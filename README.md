# NeuSim: An Open-source Simulator Framework for NPUs

## Overview

NeuSim is a simulator framework for modeling the performance and power behaviors of neural processing units (NPUs) when running machine learning workloads.

### Neural Processing Unit 101

![NPU Architecture](assets/npu_arch.svg)

An NPU chip consists of systolic arrays (SAs) for matrix multiplications and SIMD vector units (VUs) for generic vector operations. Each chip has an off-chip high-bandwidth memory (HBM) to store the ML model weights and input/output data, and an on-chip SRAM to exploit data locality and hide HBM access latency. A direct memory access (DMA) engine performs asynchronous memory copy between the HBM and SRAM. Multiple NPU chips can be connected via high-speed inter-chip interconnect (ICI) links, which form an NPU pod. A pod is typically arranged as a 2D/3D torus, which is optimized for allreduce bandwidth. The DMA engine performs remote DMA (RDMA) operations to access another chip’s HBM or SRAM.

### 🔥🔥🔥Key Features of NeuSim

![NeuSim Design](assets/simulator_design.svg)

NeuSim features:
- **Detailed performance modeling**: NeuSim models each comonent (e.g., systolic array, vector unit, on-chip SRAM, HBM, ICI) on an NPU chip and reports rich statistics for each tensor operator (e.g., execution time, FLOPS, memory traffic). It helps chip architects and system designers identify microarchitectural bottlenecks (e.g., SA-bound, VU-bound, HBM-bound).
- **Power, energy, and carbon modeling**: NeuSim models the static/dynamic power and energy consumption of each component on an NPU chip. It also models the embodied and operational carbon emissions.
- **Flexibility**: NeuSim can be invoked at different levels of granularity, including single operator simulation, end-to-end DNN model simulation, and batch simulations for design space explorations. This provides the flexibility to users with different needs.
- **Support for popular DNN models**: NeuSim takes the model graph definition as an input. It supports various popular DNN architectures, including LLMs (e.g., Llama, DeepSeek), recommendation models (e.g., DLRM), and stable diffusion models (e.g., DiT-XL, GLIGEN).
- **Multi-chip simulation**: NeuSim supports simulating multi-chip systems with different parallelism strategies (e.g., tensor parallelism, pipeline parallelism, data parallelism, expert parallelism).
- **Scalability**: A typical use case of NeuSim is design space exploration: sweeping over millions of NPU hardware configurations (e.g., number of chips) and software parameters (e.g., batch size, parallelism config) to find the "optimal" setting. NeuSim automatically parallelizes simulation jobs across multiple machines using Ray to speed up large-scale design space explorations.
- **Advanced features**: NeuSim models advanced architectural features such as power gating and dynamic voltage and frequency scaling (DVFS) to help chip architects explore the trade-offs between performance, power, and energy efficiency.


## Installation

1. [Install Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install#linux-terminal-installer) (skip if you already have conda installed).

2. NeuSim is installed as a python package. Create a conda environment and install NeuSim with `pip`:
   ```bash
   conda create --name neusim python=3.12.2
   conda activate neusim
   pip install -e .
   ```
   If you want to run unit tests or contribute to the codebase, you may also install the optional development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

## Running NeuSim

NeuSim can be launched in different ways depending on the use cases, including single operator simulations, single model simulations, batch simulations for design space explorations, etc.
The `neusim/run_scripts/` directory contains several example scripts to run NeuSim simulations.

### Quick Start
1. Start ray server:
    ```bash
    ray start --head --port=6379
    ```

2. Run the example script `neusim/run_scripts/example_npusim.sh`:
   ```bash
   cd neusim/run_scripts
   ./example_npusim.sh
   ```

   You may view the progress of the test runs in the Ray dashboard (at `http://127.0.0.1:8265/` by default; may require port forwarding if you are ssh'ing onto a remote machine).

   After the script finishes with no errors, under the "Jobs" tab in the Ray dashboard, all jobs should have the "Status" column to be "SUCCEEDED".
   An output directory `results` should be created and contain the following folders:
   - `raw/`: contains the performance simulation results. This is the output of the script `run_sim.py`.
   - `raw_None/`: contains the power simulation results. This is the output of the script `energy_operator_analysis_main.py`.
   - `carbon_NoPG/dvfs_None/CI0.0624/UTIL0.6/`: contains the results of the carbon emission analysis without power gating and DVFS, with carbon intensity 0.0624 kgCO2e/kWh and NPU chip duty cycle 60%. This is the output of the script `carbon_analysis_main.py`.
   - `slo/`: contains the SLO analysis results. This is the output of the script `slo_analysis_main.py`.

The `example_npusim.sh` script invokes the core components of NeuSim to simulate different DNN models running on various NPU hardware configurations, and analyze the output statistics to find the most cost-efficient NPU configuration that meet the target performance SLOs:

- First, it invokes `run_sim.py` for performance simulations. This script is the main entry point for running a batch of performance simulations. It sweeps over all possible number of chips, batch sizes, NPU versions, and parallelism configurations for the given DNN models. It outputs the per-operator performance statistics for each configuration to CSV files, and the end-to-end statistics as well as the simulation configuration to a JSON file. The [`Operator`](neusim/npusim/frontend/Operator.py) class contains the descriptions for all the statistics in the CSV files. This script will launch multiple Ray tasks to parallelize the simulation jobs.
- Next, it invokes `energy_operator_analysis_main.py` to run power simulations. This script reads the performance statistics generated by `run_sim.py` and computes the power and energy consumption for each operator based on the NPU hardware configuration, power gating, and DVFS settings. (Note: it is possible to integrate the power simulation into `run_sim.py`, but we separate them here for modularity and flexibility.)
- After that, it invokes `carbon_analysis_main.py` to run carbon footprint analysis and further aggregate the simulation statistics. This script reads the power and energy statistics generated by `energy_operator_analysis_main.py` and computes the carbon emissions based on the datacenter carbon intensity and NPU chip duty cycle.
- Finally, it invokes `slo_analysis_main.py`. This script analyzes the output of previous steps to find the optimal NPU configurations that meet the target SLOs (e.g., request latency for inference workloads).

A more comprehensive experiment script `run_power_gating.sh` demonstrates how to run simulations with different power gating strategies. It has the same structure as `example_npusim.sh`, but includes more models, NPU versions, and various power gating configurations.

### Customizing Simulation Parameters

#### Output Directory
Most scripts under `neusim/run_scripts` should have the `--output_dir` argument.

#### Performance Simulation Parameters
The user can specify the NPU hardware configuration and the model architecture of the simulation by creating new configuration files under `configs/`.
We provide a set of pre-defined configurations in the `configs` directory:
- `configs/chips/`: contains the NPU chip parameters, such as the number of SAs, VUs, core frequency, HBM bandwidth, on-chips SRAM size, etc.
- `configs/models/`: contains the model architecture parameters as well as the parallelism configurations. We currently support LLMs (Llama and DeepSeek), DLRM, DiT-XL, and GLIGEN. See [Defining New DNN Model Architectures](#defining-new-dnn-model-architectures) for more details on how to add support for new models.
- `configs/systems/`: contains the system-level parameters, including the datacenter power usage efficiency (PUE) and carbon intensity used for carbon emission analysis.

The script `neusim/run_scripts/run_sim.py` automatically supports new configuration files added to these directories, as long as the file names follow the existing naming conventions:
- `--models`: specify the model names. For example, if the user adds a new model configuration file `configs/models/llama4-17b.json`, the user can specify `--models="llama4-17b"` to run simulations for this model.
- `--versions`: specify the NPU chip versions. For example, if the user adds a new chip configuration file `configs/chips/tpuv7.json`, the user can specify `--versions="7"` to run simulations for this NPU version.

#### Power Simulation Parameters
The power gating parameters are defined in `neusim/npusim/frontend/power_analysis_lib.py`. The user can modify the `get_power_gating_config()` function to add new power gating configurations, including power gating wake-up cycles and power gating policies for each component.

The scripts `neusim/run_scripts/energy_operator_analysis_main.py` and `neusim/run_scripts/carbon_analysis_main.py` can be invoked withcommand line arguments.
The `--help` option shows all available options. To perform sensitivity study for power gating parameters, these two scripts support overriding the default power gating configurations via the `--power_gating_strategy` flag as follows:
- `NoPG`: no power gating.
- `Ideal`: ideal power gating with instruction-level temporal granularity and PE/ALU-level spatial granularity. This should result in the most power savings.
- `Full`: Same as `Ideal` but with non-zero power-gating factor (power_level_factors) and delay cycles.
- `<base_config>_vary_Vth_<value>_<value_sram>`: vary Vth_low (voltage when logic is power gated) and Vth_sram (voltage when SRAM cells are power gated) for sensitivity analysis. The values are the percentage over Vdd.
- `<base_config>_vary_PG_delay_<value>`: vary power gating wake-up delay for sensitivity analysis. The value is specified as the ratio over base config.

See `neusim/run_scripts/run_power_gating_sensitivity_analysis.sh` for examples of how to specify different power gating strategies via the `--power_gating_strategy` flag.
See `neusim/npusim/frontend/power_analysis_lib.py:get_power_gating_config()` for how these parameters are being handled by NeuSim internally.

### Running a Single Tensor Operator
Please see `neusim/run_scripts/run_single_op_main.py` for an example of how to run a single tensor operator simulation. This script is helpful for analyzing a specific operator of interest rather than simulating the entire DNN model.

### Running a Single Experiment
If the user wants more control over the simulation parameters, such as customizing the chip configs, model configs, and specifying the batch size and parallelism config search space, the best way is to write a custom script that invokes the `neusim.npusim.frontend` module directly.
See [`run_scripts/single_model_example.ipynb`](neusim/run_scripts/single_model_example.ipynb) for an example of creating simulation configurations and running a single experiment.

### Defining New DNN Model Architectures
We currently support LLMs (see `neusim/npusim/frontend/llm_ops_generator.py`), DLRM (see `neusim/npusim/frontend/dlrm_ops_generator.py`), DiT-XL (see `neusim/npusim/frontend/dit_ops_generator.py`), and GLIGEN (see `neusim/npusim/frontend/gligen_ops_generator.py`). Variants of these models (such as changing the number of layers or hidden dimensions) can be created by adding new configuration files in the `configs/models` directory.

To add support for new model architectures, the user needs to implement a new model generator class in `neusim/npusim/frontend` to reflect the model's execution graph. Many commonly used operators such as GEMM, Conv, and LayerNorm are implemented in `neusim/npusim/backend/npusim_lib.py`. Please refer to the existing model generator classes for examples on how to call these operators and implement new model generators.
See [`run_scripts/new_model_example.ipynb`](neusim/run_scripts/new_model_example.ipynb) for an example of adding a new model generator class and running simulations for the new model.

## Running on a Cluster

To scale out the simulator on multiple machines, we need to set up a shared storage directory and configure the Ray cluster. The instructions below shows an example of setting up a shared NFS directory and configuring a Ray cluster.

1. The NFS server can be any node in the cluster (preferably the head node).
    To set up NFS directory, run:
    ```bash
    sudo apt install nfs-kernel-server
    sudo mkdir -p /mnt/[npusim_nfs_share]
    sudo chown nobody:nogroup /mnt/[npusim_nfs_share]
    sudo chmod 777 /mnt/[npusim_nfs_share]
    echo "/mnt/[npusim_nfs_share] *(rw,sync,no_subtree_check)" | sudo tee -a /etc/exports
    sudo exportfs -a
    sudo systemctl restart nfs-kernel-server
    ```
2. On each worker node, mount the NFS directory:
    ```bash
    sudo apt install nfs-common
    sudo mkdir -p /mnt/npusim_nfs_share
    sudo mount -t nfs [head_node_ip]:/mnt/[npusim_nfs_share] /mnt/[npusim_nfs_share]
    ```

3. The GitHub repository should be cloned inside the shared NFS directory to ensure all nodes have access to the codebase.

4. The python package `neusim` must be installed on all nodes.

5. Launch ray runtime on the head node with the `neusim` conda environment:
    ```bash
    conda activate neusim
    ray start --head --port=6379
    ```
6. Finally, start the ray runtime on each worker node with the `neusim` conda environment:
    ```bash
    conda activate neusim
    ray start --address='[head_node_ip]:6379'
    ```

    You may verify all nodes are connected to the Ray cluster by running:
    ```bash
    ray status
    ```

    Alternatively, the "Cluster" tab in the Ray dashboard also shows the status of all nodes in the cluster.

7. The provided scripts under `neusim/run_scripts` can be launched on any node. Assuming we launch the scripts from the head node. Make sure the path uses the NFS shared directory, not the local path, as this path will be used by other nodes in the cluster.

8. Run the test script to verify the setup:
   ```bash
   cd /mnt/[npusim_nfs_share]/.../neusim/run_scripts
   ./example_npusim.sh
   ```

   The test script will run the same tests as in the single machine setup, but Ray will automatically distribute the tasks across all nodes.

9. Other experiment scripts can be launched in the same way as in the single machine setup, but make sure to use the NFS shared directory.


## Testing

### Running Unit Tests
We use `unittest` and `pytest` for unit testing. To run all tests, execute the following command under the repo's root directory:
```bash
pytest
```
`pytest` plugins can also be used. For example, to generate a code coverage report, run:
```bash
pytest --cov=.
```

### Writing Unit Tests
All tests covering a certain module should be placed under the `tests` directory under that module.
All test files should be named with the `test_*.py` format. If a unit test requires inputs from files, the input files should be placed under the `tests` directory (preferrably inside a sub-directory of `tests`). See `neusim/backend/tests` for examples.

## Citation

Please consider citing us if you find NeuSim useful in your research:

If you use the power modeling features, please cite:
```bibtex
@inproceedings{regate:micro25,
author = {Xue, Yuqi and Huang, Jian},
title = {ReGate: Enabling Power Gating in Neural Processing Units},
year = {2025},
url = {https://doi.org/10.1145/3725843.3756038},
booktitle = {Proceedings of the 58th IEEE/ACM International Symposium on Microarchitecture},
address = {Seoul, Korea},
series = {MICRO '25}
}
```

If you use the performance modeling features, please cite:
```bibtex
@inproceedings{neu10:micro24,
author = {Xue, Yuqi and Liu, Yiqi and Nai, Lifeng and Huang, Jian},
title = {Hardware-Assisted Virtualization of Neural Processing Units for Cloud Platforms},
year = {2024},
url = {https://doi.org/10.1109/MICRO61859.2024.00011},
booktitle = {Proceedings of the 2024 57th IEEE/ACM International Symposium on Microarchitecture},
address = {Austin, TX, USA},
series = {MICRO '24}
}

@inproceedings{v10:isca23,
author = {Xue, Yuqi and Liu, Yiqi and Nai, Lifeng and Huang, Jian},
title = {V10: Hardware-Assisted NPU Multi-tenancy for Improved Resource Utilization and Fairness},
year = {2023},
url = {https://doi.org/10.1145/3579371.3589059},
booktitle = {Proceedings of the 50th Annual International Symposium on Computer Architecture},
address = {Orlando, FL, USA},
series = {ISCA '23}
}

@inproceedings{neucloud:hotos23,
author = {Xue, Yuqi and Liu, Yiqi and Huang, Jian},
title = {System Virtualization for Neural Processing Units},
year = {2023},
url = {https://doi.org/10.1145/3593856.3595912},
booktitle = {Proceedings of the 19th Workshop on Hot Topics in Operating Systems},
address = {Providence, RI, USA},
series = {HotOS '23}
}
```
