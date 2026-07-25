from __future__ import annotations

import random
from collections.abc import Callable
from copy import deepcopy
from unittest.mock import patch

import numpy as np
import pytest

from neusim.configs.models.ModelConfig import ModelConfig
from neusim.npusim.frontend import dvfs_optimizer
from neusim.npusim.frontend.Operator import (
    ComponentDVFSConfig,
    DVFSConfig,
    DVFSPolicy,
    Operator,
)

FitnessFunction = Callable[..., tuple[np.ndarray, np.ndarray, np.ndarray]]
OffspringFunction = Callable[..., None]


def _assert_numpy_rng_state_equal(left: tuple, right: tuple) -> None:
    assert left[0] == right[0]
    np.testing.assert_array_equal(left[1], right[1])
    assert left[2:] == right[2:]


def _assert_float_arrays_bitwise_equal(
    expected: np.ndarray,
    actual: np.ndarray,
) -> None:
    np.testing.assert_array_equal(expected.view(np.uint64), actual.view(np.uint64))


def _make_case(
    *,
    population_size: int,
    num_genes: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    generator = np.random.RandomState(seed + 10_000)
    counts = generator.randint(1, 12, size=num_genes).astype(np.int32)
    population = np.empty((population_size, num_genes), dtype=np.int32)
    for gene_index, count in enumerate(counts):
        population[:, gene_index] = generator.randint(
            0,
            count,
            size=population_size,
        )
    population[0] = 0
    population[-1] = counts - 1

    max_points = int(counts.max())
    times = np.empty((num_genes, max_points), dtype=np.float64)
    energies = np.empty((num_genes, max_points), dtype=np.float64)
    for gene_index, count in enumerate(counts):
        base_time = generator.uniform(1e-6, 1e12)
        base_energy = generator.uniform(1e-12, 1e3)
        times[gene_index, :count] = base_time * generator.uniform(
            0.5,
            3.0,
            size=count,
        )
        energies[gene_index, :count] = base_energy * generator.uniform(
            0.5,
            3.0,
            size=count,
        )
        times[gene_index, count:] = np.inf
        energies[gene_index, count:] = np.inf
    scores = generator.lognormal(size=population_size)
    return population, counts, times, energies, scores


def _prepare_next_generation(
    population: np.ndarray,
    scores: np.ndarray,
    elitism_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    population_size = len(population)
    score_sum = scores.sum()
    if score_sum <= 0:
        probabilities = np.ones(population_size) / population_size
    else:
        probabilities = scores / score_sum
    new_population = np.zeros_like(population)
    elite_indices = np.argsort(scores)[-elitism_count:]
    for elite_slot, population_index in enumerate(elite_indices):
        new_population[elite_slot] = population[population_index]
    return new_population, probabilities


@pytest.mark.parametrize("seed", range(8))
@pytest.mark.parametrize(
    ("population_size", "num_genes", "elitism_count"),
    [(6, 2, 1), (7, 3, 2), (8, 19, 3), (11, 64, 2)],
)
@pytest.mark.parametrize(
    ("crossover_prob", "mutation_prob"),
    [(0.0, 0.0), (0.0, 1.0), (1.0, 0.0), (1.0, 1.0), (0.73, 0.19)],
)
def test_deferred_mutation_preserves_population_and_both_rng_states(
    seed: int,
    population_size: int,
    num_genes: int,
    elitism_count: int,
    crossover_prob: float,
    mutation_prob: float,
) -> None:
    population, counts, _times, _energies, scores = _make_case(
        population_size=population_size,
        num_genes=num_genes,
        seed=seed,
    )
    reference_population, probabilities = _prepare_next_generation(
        population,
        scores,
        elitism_count,
    )
    exact_population = reference_population.copy()
    reference_rng = random.Random(seed)
    exact_rng = random.Random(seed)
    reference_numpy_rng = np.random.RandomState(seed)
    exact_numpy_rng = np.random.RandomState(seed)

    common = {
        "population": population,
        "probabilities": probabilities,
        "num_pareto_points": counts,
        "crossover_prob": crossover_prob,
        "mutation_prob": mutation_prob,
        "elitism_count": elitism_count,
    }
    dvfs_optimizer._fill_ga_offspring_scalar_reference(
        **common,
        new_population=reference_population,
        rng=reference_rng,
        np_rng=reference_numpy_rng,
    )
    dvfs_optimizer._fill_ga_offspring_scalar_exact(
        **common,
        new_population=exact_population,
        rng=exact_rng,
        np_rng=exact_numpy_rng,
    )

    np.testing.assert_array_equal(reference_population, exact_population)
    assert reference_rng.getstate() == exact_rng.getstate()
    _assert_numpy_rng_state_equal(
        reference_numpy_rng.get_state(),
        exact_numpy_rng.get_state(),
    )



@pytest.mark.parametrize(
    ("initial_value", "step", "upper_bound"),
    [
        (np.iinfo(np.int32).max, 1, np.iinfo(np.int32).max),
        (np.iinfo(np.int32).max - 1, 1, np.iinfo(np.int32).max),
        (0, -1, np.iinfo(np.int32).max),
    ],
)
def test_collected_mutations_promote_before_int32_boundary_clipping(
    initial_value: int,
    step: int,
    upper_bound: int,
) -> None:
    expected = np.array([initial_value], dtype=np.int32)
    expected[0] = max(0, min(upper_bound, expected[0] + step))
    actual = np.array([initial_value], dtype=np.int32)

    dvfs_optimizer._apply_collected_ga_mutations(
        actual,
        [0],
        [step],
        np.array([upper_bound], dtype=np.int64),
    )

    np.testing.assert_array_equal(expected, actual)


def test_deferred_mutation_preserves_int32_max_upper_bound() -> None:
    class AlwaysPositiveMutation:
        @staticmethod
        def random() -> float:
            return 0.0

        @staticmethod
        def choice(values: list[int]) -> int:
            assert values == [-1, 1]
            return 1

        @staticmethod
        def randint(_lower: int, _upper: int) -> int:
            raise AssertionError("zero crossover probability must skip randint")

    int32_max = np.iinfo(np.int32).max
    population = np.full((3, 2), int32_max, dtype=np.int32)
    counts = np.full(2, int32_max + 1, dtype=np.int64)
    scores = np.ones(3)
    reference, probabilities = _prepare_next_generation(population, scores, 1)
    exact = reference.copy()

    common = {
        "population": population,
        "probabilities": probabilities,
        "num_pareto_points": counts,
        "crossover_prob": 0.0,
        "mutation_prob": 1.0,
        "elitism_count": 1,
    }
    dvfs_optimizer._fill_ga_offspring_scalar_reference(
        **common,
        new_population=reference,
        rng=AlwaysPositiveMutation(),
        np_rng=np.random.RandomState(17),
    )
    dvfs_optimizer._fill_ga_offspring_scalar_exact(
        **common,
        new_population=exact,
        rng=AlwaysPositiveMutation(),
        np_rng=np.random.RandomState(17),
    )

    np.testing.assert_array_equal(reference, exact)
    assert np.all(exact == int32_max)

@pytest.mark.parametrize("seed", range(10))
@pytest.mark.parametrize("shape", [(5, 2), (8, 17), (31, 257)])
@pytest.mark.parametrize("batch_size", [1, 7, 32, 64])
def test_batched_fitness_is_bit_exact(
    seed: int,
    shape: tuple[int, int],
    batch_size: int,
) -> None:
    population_size, num_genes = shape
    population, _counts, times, energies, _scores = _make_case(
        population_size=population_size,
        num_genes=num_genes,
        seed=seed,
    )
    orig_total_time = float(np.sum(times[:, 0]))
    baseline_total_energy = float(np.sum(energies[:, 0]))
    expected = dvfs_optimizer._evaluate_ga_population_scalar_reference(
        population,
        times,
        energies,
        orig_total_time,
        baseline_total_energy,
        0.73,
    )
    actual = dvfs_optimizer._evaluate_ga_population_scalar_exact(
        population,
        times,
        energies,
        orig_total_time,
        baseline_total_energy,
        0.73,
        batch_size=batch_size,
    )
    for expected_array, actual_array in zip(expected, actual, strict=True):
        _assert_float_arrays_bitwise_equal(expected_array, actual_array)


def _run_generations(
    initial_population: np.ndarray,
    counts: np.ndarray,
    times: np.ndarray,
    energies: np.ndarray,
    *,
    seed: int,
    generations: int,
    fitness: FitnessFunction,
    offspring: OffspringFunction,
) -> dict[str, object]:
    population = initial_population.copy()
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)
    orig_total_time = float(np.sum(times[:, 0]))
    baseline_total_energy = float(np.sum(energies[:, 0]))
    best_score = -np.inf
    best_individual = np.zeros(population.shape[1], dtype=np.int32)
    best_time = 0.0
    best_energy = np.inf
    evaluations = []

    for _ in range(generations):
        scores, all_times, all_energies = fitness(
            population,
            times,
            energies,
            orig_total_time,
            baseline_total_energy,
            0.73,
        )
        evaluations.append(
            (
                population.copy(),
                scores.copy(),
                all_times.copy(),
                all_energies.copy(),
            )
        )
        for individual_index in range(len(population)):
            score = scores[individual_index]
            if score > best_score:
                best_score = score
                best_individual = population[individual_index].copy()
                best_time = all_times[individual_index]
                best_energy = all_energies[individual_index]

        new_population, probabilities = _prepare_next_generation(
            population,
            scores,
            elitism_count=2,
        )
        offspring(
            population,
            new_population,
            probabilities,
            counts,
            rng,
            np_rng,
            crossover_prob=0.81,
            mutation_prob=0.23,
            elitism_count=2,
        )
        population = new_population

    scores, all_times, all_energies = fitness(
        population,
        times,
        energies,
        orig_total_time,
        baseline_total_energy,
        0.73,
    )
    evaluations.append(
        (
            population.copy(),
            scores.copy(),
            all_times.copy(),
            all_energies.copy(),
        )
    )
    for individual_index in range(len(population)):
        score = scores[individual_index]
        if score > best_score:
            best_score = score
            best_individual = population[individual_index].copy()
            best_time = all_times[individual_index]
            best_energy = all_energies[individual_index]

    return {
        "population": population,
        "best_score": best_score,
        "best_individual": best_individual,
        "best_time": best_time,
        "best_energy": best_energy,
        "evaluations": evaluations,
        "python_rng_state": rng.getstate(),
        "numpy_rng_state": np_rng.get_state(),
    }


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("population_size", [7, 8, 13])
def test_multiple_generations_preserve_every_evaluation_best_and_rng_state(
    seed: int,
    population_size: int,
) -> None:
    population, counts, times, energies, _scores = _make_case(
        population_size=population_size,
        num_genes=83,
        seed=seed,
    )
    common = {
        "initial_population": population,
        "counts": counts,
        "times": times,
        "energies": energies,
        "seed": seed,
        "generations": 7,
    }
    reference = _run_generations(
        **common,
        fitness=dvfs_optimizer._evaluate_ga_population_scalar_reference,
        offspring=dvfs_optimizer._fill_ga_offspring_scalar_reference,
    )
    exact = _run_generations(
        **common,
        fitness=dvfs_optimizer._evaluate_ga_population_scalar_exact,
        offspring=dvfs_optimizer._fill_ga_offspring_scalar_exact,
    )

    np.testing.assert_array_equal(reference["population"], exact["population"])
    np.testing.assert_array_equal(
        reference["best_individual"],
        exact["best_individual"],
    )
    for key in ("best_score", "best_time", "best_energy"):
        assert np.float64(reference[key]).tobytes() == np.float64(exact[key]).tobytes()

    for expected_evaluation, actual_evaluation in zip(
        reference["evaluations"],
        exact["evaluations"],
        strict=True,
    ):
        np.testing.assert_array_equal(expected_evaluation[0], actual_evaluation[0])
        for expected_array, actual_array in zip(
            expected_evaluation[1:],
            actual_evaluation[1:],
            strict=True,
        ):
            _assert_float_arrays_bitwise_equal(expected_array, actual_array)
    assert reference["python_rng_state"] == exact["python_rng_state"]
    _assert_numpy_rng_state_equal(
        reference["numpy_rng_state"],
        exact["numpy_rng_state"],
    )


@pytest.mark.parametrize("raw", ["0", "-2", "1.5", "not-an-integer"])
def test_exact_batch_environment_override_rejects_invalid_values(
    monkeypatch: pytest.MonkeyPatch,
    raw: str,
) -> None:
    monkeypatch.setenv("DVFS_GA_EXACT_BATCH_SIZE", raw)
    with pytest.raises(ValueError, match="DVFS_GA_EXACT_BATCH_SIZE"):
        dvfs_optimizer.configure_dvfs_c_with_degradation(
            [],
            ModelConfig(model_type="test", output_file_path=""),
            DVFSConfig(policy=DVFSPolicy.DVFS_C),
        )


def _configured_point(
    name: str,
    execution_time_ns: int,
    energy_j: float,
    marker: int,
) -> Operator:
    point = Operator(name=name, description="prefill")
    point.stats.execution_time_ns = execution_time_ns
    point.stats.static_energy_other_J = energy_j
    component = ComponentDVFSConfig(
        policy=DVFSPolicy.DVFS_C,
        voltage_V=0.5 + marker / 100.0,
        frequency_GHz=1.8 - marker / 100.0,
    )
    for field in (
        "dvfs_sa",
        "dvfs_vu",
        "dvfs_sram",
        "dvfs_hbm_mc",
        "dvfs_hbm_die",
        "dvfs_hbm_io",
        "dvfs_ici_mc",
        "dvfs_ici_phy",
    ):
        setattr(point, field, component.model_copy())
    return point


def _end_to_end_fixture() -> tuple[list[Operator], list[list[Operator]]]:
    ops = [
        _configured_point("op-a", 100, 9.0, 0),
        _configured_point("op-b", 130, 12.0, 10),
    ]
    points = [
        [
            _configured_point("op-a", 100, 9.0, 1),
            _configured_point("op-a", 112, 6.8, 2),
            _configured_point("op-a", 128, 5.2, 3),
        ],
        [
            _configured_point("op-b", 130, 12.0, 11),
            _configured_point("op-b", 148, 8.7, 12),
            _configured_point("op-b", 175, 6.4, 13),
        ],
    ]
    return ops, points


def _run_end_to_end(*, use_reference: bool) -> dict[str, object]:
    ops, points = _end_to_end_fixture()
    config = ModelConfig(model_type="test", output_file_path="")
    dvfs_config = DVFSConfig(
        policy=DVFSPolicy.DVFS_C,
        performance_degradation_percentage=0.25,
    )
    traces: list[dict] = []
    timing: dict[str, object] = {}
    exact_fitness = dvfs_optimizer._evaluate_ga_population_scalar_exact
    observed_batch_sizes: list[int] = []

    def reference_fitness(*args, batch_size, **kwargs):
        assert batch_size == 2
        return dvfs_optimizer._evaluate_ga_population_scalar_reference(
            *args,
            **kwargs,
        )

    def tracked_exact_fitness(*args, batch_size, **kwargs):
        observed_batch_sizes.append(batch_size)
        return exact_fitness(*args, batch_size=batch_size, **kwargs)

    fitness_side_effect = (
        reference_fitness if use_reference else tracked_exact_fitness
    )
    patches = [
        patch.dict(
            "os.environ",
            {
                "DVFS_GA_VECTORIZED": "0",
                "DVFS_GA_EXACT_BATCH_SIZE": "2",
            },
        ),
        patch.object(
            dvfs_optimizer,
            "_evaluate_ga_population_scalar_exact",
            side_effect=fitness_side_effect,
        ),
        patch.object(
            dvfs_optimizer,
            "_save_search_trace",
            side_effect=lambda _config, _dvfs, _ops, trace: traces.append(
                deepcopy(trace)
            ),
        ),
    ]
    if use_reference:
        patches.append(
            patch.object(
                dvfs_optimizer,
                "_fill_ga_offspring_scalar_exact",
                side_effect=dvfs_optimizer._fill_ga_offspring_scalar_reference,
            )
        )

    with patches[0], patches[1], patches[2]:
        if use_reference:
            with patches[3]:
                configured = dvfs_optimizer.configure_dvfs_c_with_degradation(
                    ops,
                    config,
                    dvfs_config,
                    population_size=7,
                    max_generations=5,
                    crossover_prob=0.73,
                    mutation_prob=0.29,
                    elitism_count=2,
                    seed=9,
                    _precomputed_points=points,
                    timing_result=timing,
                )
        else:
            configured = dvfs_optimizer.configure_dvfs_c_with_degradation(
                ops,
                config,
                dvfs_config,
                population_size=7,
                max_generations=5,
                crossover_prob=0.73,
                mutation_prob=0.29,
                elitism_count=2,
                seed=9,
                _precomputed_points=points,
                timing_result=timing,
            )

    semantic_timing = {
        key: value for key, value in timing.items() if not key.endswith("_seconds")
    }
    return {
        "configured": [op.model_dump(mode="python") for op in configured],
        "population": dvfs_optimizer.configure_dvfs_c_with_degradation.last_population.copy(),
        "best_individual": dvfs_optimizer.configure_dvfs_c_with_degradation.last_best_individual.copy(),
        "best_energy": dvfs_optimizer.configure_dvfs_c_with_degradation.last_best_energy_J,
        "trace": traces,
        "timing": semantic_timing,
        "batch_sizes": observed_batch_sizes,
    }


def test_configure_dvfs_c_exact_path_matches_reference_end_to_end() -> None:
    reference = _run_end_to_end(use_reference=True)
    exact = _run_end_to_end(use_reference=False)

    assert reference["configured"] == exact["configured"]
    np.testing.assert_array_equal(reference["population"], exact["population"])
    np.testing.assert_array_equal(
        reference["best_individual"],
        exact["best_individual"],
    )
    assert np.float64(reference["best_energy"]).tobytes() == np.float64(
        exact["best_energy"]
    ).tobytes()
    assert reference["trace"] == exact["trace"]
    assert reference["timing"] == exact["timing"]
    assert exact["batch_sizes"] == [2] * 6
    scalar_trace = exact["trace"][0]
    assert scalar_trace["ga_execution_mode"] == "scalar_exact_batched_ltr"
    assert scalar_trace["ga_exact_batch_size"] == 2
    assert scalar_trace["ga_exact_batch_size_env"] == "DVFS_GA_EXACT_BATCH_SIZE"
    assert exact["timing"]["ga_execution_mode"] == "scalar_exact_batched_ltr"
    assert exact["timing"]["ga_exact_batch_size"] == 2
    assert exact["timing"]["ga_exact_batch_size_env"] == "DVFS_GA_EXACT_BATCH_SIZE"


def test_vectorized_opt_in_remains_separate_from_exact_helpers() -> None:
    ops, points = _end_to_end_fixture()
    traces: list[dict] = []
    timing: dict[str, object] = {}
    with (
        patch.dict(
            "os.environ",
            {
                "DVFS_GA_VECTORIZED": "1",
                "DVFS_GA_EXACT_BATCH_SIZE": "invalid-but-unused",
            },
        ),
        patch.object(
            dvfs_optimizer,
            "_evaluate_ga_population_scalar_exact",
            side_effect=AssertionError("exact evaluator must be bypassed"),
        ),
        patch.object(
            dvfs_optimizer,
            "_fill_ga_offspring_scalar_exact",
            side_effect=AssertionError("exact offspring path must be bypassed"),
        ),
        patch.object(
            dvfs_optimizer,
            "_save_search_trace",
            side_effect=lambda _config, _dvfs, _ops, trace: traces.append(
                deepcopy(trace)
            ),
        ),
    ):
        configured = dvfs_optimizer.configure_dvfs_c_with_degradation(
            ops,
            ModelConfig(model_type="test", output_file_path=""),
            DVFSConfig(
                policy=DVFSPolicy.DVFS_C,
                performance_degradation_percentage=0.25,
            ),
            population_size=6,
            max_generations=2,
            elitism_count=2,
            seed=4,
            _precomputed_points=points,
            timing_result=timing,
        )

    assert configured
    assert traces[0]["ga_execution_mode"] == "vectorized_non_bit_exact"
    assert traces[0]["ga_exact_batch_size"] is None
    assert traces[0]["ga_exact_batch_size_env"] == "DVFS_GA_EXACT_BATCH_SIZE"
    assert timing["ga_execution_mode"] == "vectorized_non_bit_exact"
    assert timing["ga_exact_batch_size"] is None
    assert timing["ga_exact_batch_size_env"] == "DVFS_GA_EXACT_BATCH_SIZE"
