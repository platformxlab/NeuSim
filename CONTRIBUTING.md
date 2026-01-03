# Contributing to NeuSim

We welcome contributions from the community to help improve this project. Please take a moment to review this document before submitting any issues or pull requests.

## Codebase Structure

The NeuSim codebase is organized as follows:

- **`assets/`**: Contains images and other static assets used in documentation.
- **`configs/`**: Contains JSON configuration files for NPU chips, DNN models, and system settings.
    - `chips/`: NPU chip hardware configurations.
    - `models/`: DNN model configurations.
    - `systems/`: System-level configurations.
- **`neusim/`**: The main Python package containing the simulator source code.
    - **`configs/`**: Pydantic models for validating configuration files.
        - `chips/`: Chip configuration classes.
        - `models/`: Model configuration classes.
        - `systems/`: System configuration classes.
    - **`npusim/`**: The core simulator logic.
        - `backend/`: Backend performance simulation logic.
        - `frontend/`: Ops generators (defining DNN model graphs) and power modeling.
    - **`run_scripts/`**: Scripts for running simulations, analysis, and examples.
    - **`xla_hlo_parser/`**: XLA-style program parser and analyzer framework. Used in NeuSim backend for analyzing operator semantics. This module may be refactored into the backend in future releases.

## Issues

We use GitHub Issues to track bugs, feature requests, and other tasks.

### Creating an Issue
1. **Search existing issues**: Before creating a new issue, please search the existing issues to see if your problem or feature request has already been reported.
2. **Use a clear title**: Provide a concise and descriptive title for your issue.
3. **Provide details**:
    - For **bugs**: Describe the issue, provide steps to reproduce it, and include any relevant error messages or logs.
    - For **feature requests**: Describe the proposed feature, its use case, and why it would be beneficial.
4. **Labeling**: If possible, add appropriate labels to your issue (e.g., `bug`, `enhancement`, `documentation`).

## Pull Requests

We welcome pull requests (PRs) for bug fixes, new features, and documentation improvements.

### Submitting a Pull Request
1. **Fork the repository**: Create a fork of the NeuSim repository to your own GitHub account.
2. **Create a branch**: Create a new branch for your changes. Use a descriptive name (e.g., `fix-bug-xyz`, `add-feature-abc`).
3. **Make changes**: Implement your changes. We recommend making small, focused commits with clear messages. We will add code formatting checks using `black` in the future.
4. **Run tests**: If you added new functionality, please add corresponding tests. Ensure that all existing tests pass before submitting your PR.
    - You can install development dependencies with `pip install -e ".[dev]"`.
    - Run all tests under the root directory using `pytest --cov=.`.
    - For coverage, please ensure that your changes maintain or improve the current test coverage. For new files, aim for at least 90% coverage. For special cases where this is not feasible, provide a justification in your PR.
5. **Update documentation**: If your changes affect the user interface or configuration, please update the relevant documentation (e.g., `README.md`).
6. **Submit the PR**: Open a pull request against the `main` branch of the NeuSim repository. Provide a clear description of your changes and reference any related issues.

### Coding Standards
- **Type Hinting**: We encourage the use of Python type hints for better code readability and maintainability.
- **Comments**: Add comments to explain complex logic or non-obvious code sections.
- We will add more detailed code style guidelines in the future.

## Code Reviews

All pull requests will be reviewed by the maintainers before being merged.

### Review Process
Please address all review comments. You can either make the requested changes or discuss why you believe the current implementation is correct.
Once the reviewers are satisfied with your changes, they will approve the PR, and it will be merged.

Thank you for contributing to NeuSim!
