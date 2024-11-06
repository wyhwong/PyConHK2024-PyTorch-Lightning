# Default functionality

This README introduces the available make commands in this project template.

## Prerequisites:
- GNU make: [https://www.gnu.org/software/make/manual/make.html](https://www.gnu.org/software/make/manual/make.html)
- Poetry: [https://python-poetry.org](https://python-poetry.org)
- Docker: [https://www.docker.com/](https://www.docker.com/)

## Run in Local Enrivonment

#### Installation

```bash
# Install poetry if needed
pip3 install poetry

# Install dependencies
make install
```

## Setting up Infrastructure (When needed)

```bash
# Build all infrastructure
make infra-build

# Start all infrastructure
make infra-up

# Stop all infrastructure
make infra-down
```

## Development

Tools used in development:
- black: [https://black.readthedocs.io/en/stable/](https://black.readthedocs.io/en/stable/)
- bandit: [https://bandit.readthedocs.io/en/latest/](https://bandit.readthedocs.io/en/latest/)
- pylint: [https://pylint.readthedocs.io/en/stable/](https://pylint.readthedocs.io/en/stable/)
- isort: [https://pycqa.github.io/isort/](https://pycqa.github.io/isort/)
- mypy: [https://mypy.readthedocs.io/en/stable/](https://mypy.readthedocs.io/en/stable/)
- radon: [https://radon.readthedocs.io/en/latest/](https://radon.readthedocs.io/en/latest/)
- pytest: [https://docs.pytest.org/en/latest/](https://docs.pytest.org/en/latest/)
- pytest-asyncio: [https://pypi.org/project/pytest-asyncio/](https://pypi.org/project/pytest-asyncio/)
- pytest-cov: [https://github.com/pytest-dev/pytest-cov](https://github.com/pytest-dev/pytest-cov)
- pytest-env: [https://pypi.org/project/pytest-env/](https://pypi.org/project/pytest-env/)
- pre-commit: [https://pre-commit.com/](https://pre-commit.com/)

```bash
# Show make help
make help

# Run test cases
make test

# Run static code analysis
make static-code-analysis

# Run test report
make test-report

# Format code with black, isort
make format

# Update dependencies
make update

# Generate UML diagram
make uml-diagram

# Show environment information used in this project
make app-info
```

## Run in Docker Environment

Dockerfile: [./deployment/Dockerfile](./deployment/Dockerfile)

#### Build and Run Docker Container

```bash
# Build Docker image
make build

# Run Docker container
make run
```

#### Stop Docker Container

```bash
# Stop Docker container
make clean
```
