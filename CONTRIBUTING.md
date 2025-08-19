# Contributing to MetagenomicsOS

Thank you for your interest in contributing to MetagenomicsOS! This document provides guidelines for contributing to this project.

## Getting Started

### Prerequisites

- Python 3.9 or higher
- uv package manager
- Git

### Setting Up Development Environment

1. Clone the repository:

git clone https://github.com/Subhadip1409/metagenomicsOS.git

2. Install dependencies:

3. Install pre-commit hooks (we'll set this up tomorrow):

## Development Workflow

### Branching Strategy

- `main` - stable, production-ready code
- `develop` - integration branch for features
- `feature/feature-name` - individual features
- `bugfix/issue-description` - bug fixes

### Making Changes

1. Create a feature branch:
   git checkout -b feature/your-feature-name

2. Make your changes and write tests
3. Run tests to ensure everything works:

4. Commit your changes with a descriptive message:
   git commit -m "feat: add new feature description"

### Pull Request Process

1. Push your branch to GitHub
2. Create a Pull Request with:

- Clear title and description
- Link to any related issues
- Screenshots if UI changes are involved

3. Ensure all CI checks pass
4. Wait for code review and address feedback

## Coding Standards

- Follow PEP 8 Python style guidelines
- Use type hints for all function parameters and return values
- Write docstrings for all public functions and classes
- Keep functions small and focused
- Write tests for new features

## Questions?

Feel free to open an issue for questions or join our discussions!
