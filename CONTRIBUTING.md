# Contributing to MetagenomicsOS

Thank you for your interest in contributing to MetagenomicsOS! We welcome all contributions.

## How to Contribute

*   **Report Bugs:** If you find a bug, please open an issue on our [GitHub issue tracker](https://github.com/Subhadip1409/metagenomicsOS/issues).
*   **Suggest Features:** Have an idea? Open a feature request issue to start a discussion.
*   **Write Code:** If you want to contribute code, please follow the development workflow below.

## Development Workflow

1.  **Fork the repository** and clone it locally.
2.  **Create a feature branch:** `git checkout -b your-feature-name`
3.  **Install dependencies.** It's highly recommended to install the `dev` and `pre-commit` dependencies:
    ```bash
    pip install -e ".[dev]"
    pre-commit install
    ```
4.  **Make your changes.** Write clean, readable code.
5.  **Run tests** to ensure your changes don't break anything: `pytest`
6.  **Commit your changes** with a descriptive message. The pre-commit hooks will automatically format and lint your code.
7.  **Push your branch** to your fork.
8.  **Open a Pull Request** to the main repository.

## Coding Standards

We use the following tools to maintain code quality. The pre-commit hooks will manage these for you.

*   **Formatter:** `black`
*   **Import Sorter:** `isort`
*   **Linter:** `flake8`
*   **Type Checker:** `mypy`

We look forward to your contributions!
