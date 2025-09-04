# Contributing to nhm-assist

Thank you for your interest in contributing to nhm-assist!

## Getting Started

### Forking Workflow

1. **Fork the repository**: Click the "Fork" button on the main repository page to create your own copy
2. **Clone your fork locally**: Clone your forked repository to your local machine
3. **Add upstream remote**: Add the original repository as an upstream remote to keep your fork synchronized
4. **Create a feature branch**: Always work on a new branch for your changes
5. **Make your changes**: Edit files and commit your changes
6. **Push to your fork**: Push your feature branch to your forked repository
7. **Submit a pull request**: Create a pull request from your fork to the main repository

### Quick Start

1. Fork the repository on the web interface
2. Clone your fork and set up remotes
3. Set up the development environment (see below)
4. Create a feature branch for your changes
5. Make your changes and test them
6. Submit a pull request

## Development Environment

1. Install [conda-forge miniforge](https://github.com/conda-forge/miniforge)

2. Fork and clone the repository:

   ```bash
   # Fork the repository on the web interface first, then:
   git clone https://code.usgs.gov/wma/hytest/YOUR-USERNAME/nhm-assist.git
   cd nhm-assist
   
   # Add the original repository as upstream
   git remote add upstream https://code.usgs.gov/wma/hytest/nhm-assist.git
   
   # Verify your remotes
   git remote -v
   ```

3. Create the conda environment:

   ```bash
   mamba env create -f environment.yaml
   mamba activate nhm
   ```

4. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## Making Changes

- Edit Python scripts in the `notebook_scripts/` directory (these are the source files)
- Edit Python modules in the `nhm_helpers/` directory
- Run `python make_notebooks.py` to generate notebook files from the scripts
- Test your changes by running the generated notebooks
- Note: Only `.py` files are tracked in git; `.ipynb` files are generated automatically

## Code Standards

- Follow PEP 8 for Python code
- Use clear, descriptive variable names
- Add docstrings to functions
- Ensure notebooks can run from start to finish

## Submitting Changes

### Step-by-Step Process

1. **Sync your fork** (if you've had it for a while):

   ```bash
   git checkout main
   git fetch upstream
   git merge upstream/main
   git push origin main
   ```

2. **Create a feature branch**:

   ```bash
   git checkout -b feature-name
   ```

3. **Make your changes and commit them**:

   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. **Run pre-commit checks**:

   ```bash
   pre-commit run --all-files
   ```

5. **Push to your fork**:

   ```bash
   git push origin feature-name
   ```

6. **Submit a pull request**: Go to your fork on the web interface and create a pull request to the main repository

### Pull Request Guidelines

- Use a clear, descriptive title
- Explain what changes you made and why
- Reference any related issues
- Ensure all tests pass and pre-commit checks are clean

## Questions?

If you have questions, please create an issue or refer to the [README.md](README.md) for more information.
