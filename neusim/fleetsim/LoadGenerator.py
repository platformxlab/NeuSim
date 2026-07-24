import csv
import math
import uuid

import numpy as np
import pandas as pd
from absl import logging

import neusim.fleetsim.util as util
from neusim.configs.workloads.LLMInferenceWorkloadConfig import (
    LLMInferenceWorkloadConfig,
    RequestPatternType,
)


class LLMRequest:
    __slots__ = [
        "id",
        "input_seqlen",
        "output_seqlen",
        "enqueue_timestamp",
        "prefill_start_timestamp",
        "prefill_end_timestamp",
        "decode_start_timestamp",
        "decode_end_timestamp",
        "current_decode_step",
        "ideal_TTFT_ns",
        "ideal_TPOT_ns",
        "prefill_energy_J",
        "decode_energy_J",
        "prefill_cost_dollars",
        "decode_cost_dollars",
    ]

    def __init__(
        self,
        input_seqlen: int,
        output_seqlen: int,
        timestamp: int | None = None,
        request_id: str | None = None,
    ):
        if input_seqlen < 1:
            raise ValueError(f"input_seqlen must be positive, got {input_seqlen}")
        if output_seqlen < 2:
            raise ValueError(
                f"output_seqlen must be at least 2, got {output_seqlen}"
            )
        if timestamp is not None and timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {timestamp}")
        self.id: str = request_id if request_id is not None else uuid.uuid4().hex
        """Unique identifier for the request."""
        self.input_seqlen: int = input_seqlen
        self.output_seqlen: int = output_seqlen
        self.enqueue_timestamp: int = timestamp if timestamp is not None else 0
        """
        Enqueue timestamp in nanoseconds.
        """
        self.prefill_start_timestamp: int = -1
        """
        Prefill start timestamp in nanoseconds.
        -1 means prefill phase has not started yet.
        """
        self.prefill_end_timestamp: int = -1
        """
        Prefill end timestamp in nanoseconds.
        -1 means prefill phase has not finished yet.
        """
        self.decode_start_timestamp: int = -1
        """
        Decode start timestamp in nanoseconds.
        -1 means decode phase has not started yet.
        """
        self.decode_end_timestamp: int = -1
        """
        Decode end timestamp in nanoseconds.
        -1 means decode phase has not finished yet.
        """

        self.current_decode_step: int = -1
        """
        Counter for the number of decoded tokens. \\
        Decode step == -1 means prefill phase is not started yet. \\
        Decode step == 0 means prefill phase is not finished yet. \\
        Decode step == 1 means prefill phase is finished but decode phase is not started yet. \\
        Decode step > 1 means decode phase is ongoing. \\
        Decode step >= self.output_seqlen means decode phase is finished.
        """

        self.ideal_TTFT_ns: int = -1
        """
        Ideal time to first token (TTFT) in nanoseconds without queuing delay or pipeline HOL blocking.
        """
        self.ideal_TPOT_ns: int = -1
        """
        Ideal time to process one token (TPOT) in nanoseconds without pipeline HOL blocking.
        """

        self.prefill_energy_J: float = 0.0
        """Total energy consumed by prefill in joules."""
        self.decode_energy_J: float = 0.0
        """Total energy consumed by all decode iterations in joules."""

        self.prefill_cost_dollars: float = 0.0
        """Total cost of prefill in dollars."""
        self.decode_cost_dollars: float = 0.0
        """Total monetary cost of all decode iterations in dollars."""

    @property
    def total_seqlen(self) -> int:
        return self.input_seqlen + self.output_seqlen

    def mark_prefill_started(self, prefill_start_timestamp: int):
        assert self.prefill_start_timestamp == -1, "Prefill phase has already started."
        assert (
            self.current_decode_step == -1
        ), "Prefill phase should not be already started."
        self.prefill_start_timestamp = prefill_start_timestamp
        self.current_decode_step = 0

    def mark_prefill_finished(self, prefill_end_timestamp: int):
        assert self.prefill_start_timestamp != -1, "Prefill phase has not started yet."
        assert self.prefill_end_timestamp == -1, "Prefill phase has already finished."
        assert self.current_decode_step == 0, "Prefill phase should be ongoing."
        self.prefill_end_timestamp = prefill_end_timestamp
        self.current_decode_step = 1

    def mark_decode_iteration_started(self, decode_start_timestamp: int):
        """
        Mark the start of a decode iteration.
        If this is the first decode iteration, it will also mark the start of the decode phase.
        """
        assert self.prefill_end_timestamp != -1, "Prefill phase has not finished yet."
        assert self.current_decode_step >= 1, "Prefill should be finished."
        if self.current_decode_step == 1:
            # first decode step
            assert (
                self.decode_start_timestamp == -1
            ), "Decode phase has already started."
            self.decode_start_timestamp = decode_start_timestamp

    def mark_decode_iteration_finished(self, timestamp: int, num_iterations: int):
        """
        Mark the end of a decode iteration.
        A new decode iteration can be started after this.
        If this is the last decode iteration, it will also mark the end of the decode phase.
        """
        assert self.decode_start_timestamp != -1, "Decode phase has not started yet."
        assert (
            self.decode_end_timestamp == -1
        ), f"Decode phase has already finished. Request: {self}"
        assert self.current_decode_step >= 1, "Decode phase should be ongoing."
        if num_iterations <= 0:
            raise ValueError(
                f"num_iterations must be positive, got {num_iterations}"
            )
        remaining_iterations = self.output_seqlen - self.current_decode_step
        if num_iterations > remaining_iterations:
            raise ValueError(
                f"Decode iteration would overshoot request {self.id}: "
                f"{num_iterations} scheduled with only {remaining_iterations} remaining."
            )
        self.current_decode_step = self.current_decode_step + num_iterations
        if self.is_decode_finished():
            # finish decode phase
            self.decode_end_timestamp = timestamp
            logging.debug(
                "Request %s finished decoding at timestamp %d", self.id, timestamp
            )

    def is_prefill_started(self) -> bool:
        return self.current_decode_step >= 0

    def is_prefill_finished(self) -> bool:
        return self.current_decode_step >= 1

    def is_decode_started(self) -> bool:
        return self.current_decode_step >= 1 and self.decode_start_timestamp != -1

    def is_decode_finished(self) -> bool:
        return self.current_decode_step >= self.output_seqlen

    def total_latency_ns(self) -> int:
        """
        Returns the total latency in nanoseconds.
        This is the time from enqueue to decode end.
        """
        return self.decode_end_timestamp - self.enqueue_timestamp

    def prefill_queuing_delay_ns(self) -> int:
        """
        Returns the queuing delay in nanoseconds.
        """
        return self.prefill_start_timestamp - self.enqueue_timestamp

    def prefill_latency_ns(self) -> int:
        """
        Returns the prefill latency in nanoseconds.
        """
        return self.prefill_end_timestamp - self.prefill_start_timestamp

    def decode_latency_ns(self) -> int:
        """
        Returns the decode latency in nanoseconds.
        This is the time from decode start to decode end.
        """
        return self.decode_end_timestamp - self.decode_start_timestamp

    def TTFT_ns(self) -> int:
        """
        Time to first token (TTFT) in nanoseconds.
        Returns the total time from enqueue to prefill end in nanoseconds.
        """
        assert self.is_prefill_finished(), "Prefill phase is not finished yet."
        return self.prefill_end_timestamp - self.enqueue_timestamp

    def TPOT_ns(self) -> int:
        """
        Time per output token (TPOT) in nanoseconds: average inter-token latency over output
        tokens 2..N. Token 1 is emitted at prefill_end, so the gap to token 2 includes the
        prefill->decode handoff queue (decode_start - prefill_end). We measure from
        prefill_end (not decode_start) so a request that waits for a decode slot is charged
        for that wait -- otherwise a decode backlog (the system failing to keep up) stays
        invisible to TPOT. For a contention-free deployment decode_start == prefill_end, so
        this is unchanged.
        """
        assert self.is_decode_finished(), "Decode phase is not finished yet."
        return math.ceil(
            (self.decode_end_timestamp - self.prefill_end_timestamp)
            / (self.output_seqlen - 1)
        )

    def decode_queuing_delay_per_iteration_ns(self) -> int:
        """
        Returns the decode queuing delay per iteration in nanoseconds.
        This is the actual TPOT_ns minus the ideal TPOT_ns.
        """
        assert self.ideal_TPOT_ns != -1, "Ideal TPOT is not set."
        return self.TPOT_ns() - self.ideal_TPOT_ns

    def __str__(self) -> str:
        return (
            f"LLMRequest(id={self.id}, input_seqlen={self.input_seqlen}, output_seqlen={self.output_seqlen}, current_decode_step={self.current_decode_step}, "
            f"enqueue_timestamp={self.enqueue_timestamp}, prefill_start_timestamp={self.prefill_start_timestamp}, "
            f"prefill_end_timestamp={self.prefill_end_timestamp}, decode_start_timestamp={self.decode_start_timestamp}, "
            f"decode_end_timestamp={self.decode_end_timestamp})"
        )

    def __repr__(self) -> str:
        return str(self)


class LoadGenerator:
    def __init__(self, config: LLMInferenceWorkloadConfig):
        self.config: LLMInferenceWorkloadConfig = config

    def generate(self) -> list[LLMRequest]:
        if self.config.request_pattern == RequestPatternType.TRACE:
            # Load requests from the trace file
            return self.load_requests_from_trace()
        elif self.config.request_pattern == RequestPatternType.SYNTHETIC:
            return self._generate_synthetic_requests()
        else:
            raise NotImplementedError(
                f"Request pattern {self.config.request_pattern} is not supported yet."
            )

    def load_requests_from_trace(self) -> list[LLMRequest]:
        if "BurstGPT" in self.config.trace_file_path:
            reqs = self._load_requests_from_trace_BurstGPT()
        else:  # Currently assuming Azure trace format by default
            reqs = self._load_requests_from_trace_Azure()
        logging.info(
            "Loaded %d requests from trace file: %s",
            len(reqs),
            self.config.trace_file_path,
        )
        if self.config.pad_seqlen_loadgen:
            # Apply padding to the sequence lengths
            for req in reqs:
                req.input_seqlen = util.pad_seqlen(
                    req.input_seqlen,
                    self.config.input_seqlen_padding_factors,
                    self.config.input_seqlen_padding_steps,
                )
                req.output_seqlen = util.pad_seqlen(
                    req.output_seqlen,
                    self.config.output_seqlen_padding_factors,
                    self.config.output_seqlen_padding_steps,
                )

        if self.config.request_rate > 1.0:
            # Scale the request rate by an integer factor while preserving the trace's
            # temporal pattern. For each request we emit N copies spread evenly across the
            # gap to the next arrival, so the arrival rate is multiplied by N at every instant
            # (the ramp/shape is preserved) instead of piling copies at the same timestamp.
            n = int(round(self.config.request_rate))
            original_reqs = sorted(reqs, key=lambda r: r.enqueue_timestamp)
            m = len(original_reqs)
            reqs = []
            for j, req in enumerate(original_reqs):
                t = req.enqueue_timestamp
                if j + 1 < m:
                    gap = original_reqs[j + 1].enqueue_timestamp - t
                elif j > 0:
                    gap = t - original_reqs[j - 1].enqueue_timestamp
                else:
                    gap = n
                gap = max(int(gap), 1)
                for k in range(n):
                    reqs.append(
                        LLMRequest(
                            input_seqlen=req.input_seqlen,
                            output_seqlen=req.output_seqlen,
                            timestamp=int(t + k * gap // n),
                            request_id=str(len(reqs)),
                        )
                    )

        logging.info(
            "Seqlen stats (padded):\nPrefill: Max=%d, Min=%d, Avg=%d\nDecode: Max=%d, Min=%d, Avg=%d",
            util.pad_seqlen(
                max([r.input_seqlen for r in reqs]),
                self.config.input_seqlen_padding_factors,
                self.config.input_seqlen_padding_steps,
            ),
            util.pad_seqlen(
                min([r.input_seqlen for r in reqs]),
                self.config.input_seqlen_padding_factors,
                self.config.input_seqlen_padding_steps,
            ),
            util.pad_seqlen(
                math.ceil(sum([r.input_seqlen for r in reqs]) / len(reqs)),
                self.config.input_seqlen_padding_factors,
                self.config.input_seqlen_padding_steps,
            ),
            util.pad_seqlen(
                max([r.output_seqlen for r in reqs]),
                self.config.output_seqlen_padding_factors,
                self.config.output_seqlen_padding_steps,
            ),
            util.pad_seqlen(
                min([r.output_seqlen for r in reqs]),
                self.config.output_seqlen_padding_factors,
                self.config.output_seqlen_padding_steps,
            ),
            util.pad_seqlen(
                math.ceil(sum([r.output_seqlen for r in reqs]) / len(reqs)),
                self.config.output_seqlen_padding_factors,
                self.config.output_seqlen_padding_steps,
            ),
        )

        return reqs

    def _load_requests_from_trace_Azure(self) -> list[LLMRequest]:
        requests = []
        request_limit = (
            self.config.max_num_requests
            if self.config.max_num_requests > 0 and self.config.request_rate <= 1.0
            else None
        )
        # Read the trace file using pandas
        df = pd.read_csv(
            self.config.trace_file_path,
            usecols=["TIMESTAMP", "ContextTokens", "GeneratedTokens"],
            parse_dates=["TIMESTAMP"],
            date_format="mixed",
            dayfirst=False,
            nrows=request_limit,
        )

        # Get the reference timestamp (first request)
        reference_time = df["TIMESTAMP"].iloc[0].timestamp() * 1e9

        for idx, row in enumerate(df.itertuples(index=False)):
            # Calculate the offset in nanoseconds
            timestamp_ns = row.TIMESTAMP.timestamp() * 1e9 - reference_time
            # if self.config.max_timestamp != -1 and timestamp_ns > self.config.max_timestamp:
            #     continue
            # if self.config.max_num_requests != -1 and len(requests) >= self.config.max_num_requests:
            #     break
            req = LLMRequest(
                input_seqlen=int(row.ContextTokens),
                # Generate at least 2 tokens to make sure the decode stage is triggered.
                output_seqlen=max(2, int(row.GeneratedTokens)),
                timestamp=int(timestamp_ns),
                request_id=str(idx),
            )
            requests.append(req)

        return requests

    def _load_requests_from_trace_BurstGPT(self) -> list[LLMRequest]:
        requests = []

        with open(self.config.trace_file_path) as f:
            reader = csv.DictReader(f)
            traces = list(reader)

        reference_time = int(traces[0]["Timestamp"]) * 1e9  # convert to ns

        for idx, row in enumerate(traces):
            timestamp_ns = int(row["Timestamp"]) * 1e9 - reference_time
            # if self.config.max_timestamp != -1 and timestamp_ns > self.config.max_timestamp:
            #     continue
            # if self.config.max_num_requests != -1 and len(requests) >= self.config.max_num_requests:
            #     break
            req = LLMRequest(
                input_seqlen=int(row["Request tokens"]),
                # Generate at least 2 tokens to make sure the decode stage is triggered.
                output_seqlen=max(2, int(row["Response tokens"])),
                timestamp=int(timestamp_ns),
                request_id=str(idx),
            )
            requests.append(req)

        return requests

    def _generate_synthetic_requests(self) -> list[LLMRequest]:
        cfg = self.config
        n = cfg.synthetic_num_requests
        rate = cfg.synthetic_request_rate
        rng = np.random.default_rng(cfg.synthetic_seed)

        # Generate sequence lengths
        if cfg.synthetic_input_len_std > 0:
            input_lens = rng.normal(
                cfg.synthetic_input_len, cfg.synthetic_input_len_std, size=n
            )
            input_lens = np.maximum(input_lens, 1).astype(int)
        else:
            input_lens = np.full(n, cfg.synthetic_input_len, dtype=int)

        if cfg.synthetic_output_len_std > 0:
            output_lens = rng.normal(
                cfg.synthetic_output_len, cfg.synthetic_output_len_std, size=n
            )
            output_lens = np.maximum(output_lens, 2).astype(int)
        else:
            output_lens = np.full(n, cfg.synthetic_output_len, dtype=int)

        # Generate arrival timestamps (Poisson process)
        if math.isinf(rate):
            timestamps_ns = np.zeros(n, dtype=int)
        else:
            intervals = rng.exponential(1.0 / rate, size=n)
            intervals[0] = 0.0  # first request arrives at t=0
            timestamps_ns = np.cumsum(intervals)
            timestamps_ns = (timestamps_ns * 1e9).astype(int)  # convert seconds to ns

        requests = []
        for idx in range(n):
            input_seqlen = int(input_lens[idx])
            output_seqlen = int(output_lens[idx])

            if cfg.pad_seqlen_loadgen:
                input_seqlen = util.pad_seqlen(
                    input_seqlen,
                    cfg.input_seqlen_padding_factors,
                    cfg.input_seqlen_padding_steps,
                )
                output_seqlen = util.pad_seqlen(
                    output_seqlen,
                    cfg.output_seqlen_padding_factors,
                    cfg.output_seqlen_padding_steps,
                )

            req = LLMRequest(
                input_seqlen=input_seqlen,
                output_seqlen=output_seqlen,
                timestamp=int(timestamps_ns[idx]),
                request_id=str(idx),
            )
            requests.append(req)

        logging.info(
            "Generated %d synthetic requests (rate=%.2f RPS, seed=%d). "
            "Input seqlen: %d (std=%d), Output seqlen: %d (std=%d)",
            n,
            rate,
            cfg.synthetic_seed,
            cfg.synthetic_input_len,
            cfg.synthetic_input_len_std,
            cfg.synthetic_output_len,
            cfg.synthetic_output_len_std,
        )

        return requests
