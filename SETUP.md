# How to Push This to GitHub

## Option 1: GitHub CLI

```bash
# Install GitHub CLI if needed: https://cli.github.com
gh auth login
gh repo create dp-100-problems --public --description "100 Dynamic Programming problems with solutions and unit tests"
cd dp-tasks
git init
git add .
git commit -m "Initial commit: 100 DP problems with tests"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/dp-100-problems.git
git push -u origin main
```

## Option 2: GitHub Website + Git

```bash
# 1. Create repo at https://github.com/new
# 2. Then locally:
cd dp-tasks
git init
git add .
git commit -m "Initial commit: 100 DP problems with tests"
git remote add origin https://github.com/YOUR_USERNAME/REPO_NAME.git
git branch -M main
git push -u origin main
```

## Running Tests Locally

```bash
pip install pytest pytest-cov
python -m pytest tests/ -v
python -m pytest tests/ --cov=dp_solutions --cov-report=html
```
