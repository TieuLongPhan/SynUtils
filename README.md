# SynUtils

**Utils for Synthesis Planning**

SynUtils is a collection of tools designed to support the planning and execution of chemical synthesis. This repository provides computational resources and utilities aimed at enhancing the efficiency and accuracy of synthesis planning through the use of advanced algorithms and AI-driven models.

![SynUtils](Docs/Figure/synutils.png)

Our tools are tailored to assist researchers and chemists in navigating complex chemical reactions and synthesis pathways, leveraging the power of modern computational chemistry. Whether you're designing novel compounds or optimizing existing processes, SynUtils aims to provide the critical tools you need.

For more details on each utility within the repository, please refer to the documentation provided in the respective folders.

## Step-by-Step Installation Guide

1. **Python Installation:**
  Ensure that Python 3.11 or later is installed on your system. You can download it from [python.org](https://www.python.org/downloads/).

2. **Creating a Virtual Environment (Optional but Recommended):**
  It's recommended to use a virtual environment to avoid conflicts with other projects or system-wide packages. Use the following commands to create and activate a virtual environment:

  ```bash
  python -m venv synutils-env
  source synutils-env/bin/activate  
  ```
  Or Conda

  ```bash
  conda create --name synutils-env python=3.11
  conda activate synutils-env
  ```

3. **Cloning and Installing SynUtils:**
  Clone the SynUtils repository from GitHub and install it:

  ```bash
  git clone https://github.com/TieuLongPhan/SynUtils.git
  cd SynUtils
  pip install -r requirements.txt
  pip install black flake8 pytest # black for formating, flake8 for checking format, pytest for testing
  ```


## Setting Up Your Development Environment

Before you start, ensure your local development environment is set up correctly. Pull the latest version of the `main` branch to start with the most recent stable code.

```bash
git checkout main
git pull
```

## Working on New Features

1. **Create a New Branch**:  
   For every new feature or bug fix, create a new branch from the `main` branch. Name your branch meaningfully, related to the feature or fix you are working on.

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Develop and Commit Changes**:  
   Make your changes locally, commit them to your branch. Keep your commits small and focused; each should represent a logical unit of work.

   ```bash
   git commit -m "Describe the change"
   ```

3. **Run Quality Checks**:  
   Before finalizing your feature, run the following commands to ensure your code meets our formatting standards and passes all tests:

   ```bash
   ./lint.sh # Check code format
   pytest Test # Run tests
   ```

   Fix any issues or errors highlighted by these checks.

## Integrating Changes

1. **Rebase onto Staging**:  
   Once your feature is complete and tests pass, rebase your changes onto the `staging` branch to prepare for integration.

   ```bash
   git fetch origin
   git rebase origin/staging
   ```

   Carefully resolve any conflicts that arise during the rebase.

2. **Push to Your Feature Branch**:
   After successfully rebasing, push your branch to the remote repository.

   ```bash
   git push origin feature/your-feature-name
   ```

3. **Create a Pull Request**:
   Open a pull request from your feature branch to the `stagging` branch. Ensure the pull request description clearly describes the changes and any additional context necessary for review.

## Important Notes