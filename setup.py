#!/usr/bin/env python3
"""
Setup script for MentorEval package.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
try:
    long_description = (this_directory / "README.md").read_text(encoding="utf-8")
except:
    long_description = "MentorEval: A comprehensive benchmark for evaluating student exam grading systems"

# Read requirements from requirements.txt
def read_requirements():
    requirements_file = this_directory / "requirements.txt"
    if requirements_file.exists():
        with open(requirements_file, 'r', encoding='utf-8') as f:
            requirements = []
            git_dependencies = []
            for line in f:
                line = line.strip()
                # Skip empty lines and comments
                if line and not line.startswith('#'):
                    # Handle git+ URLs and regular packages
                    if line.startswith('git+'):
                        git_dependencies.append(line)
                    else:
                        requirements.append(line)
            return requirements, git_dependencies
    return [], []

# Get requirements
install_requires, git_dependencies = read_requirements()

setup(
    name="mentoreval",
    version="0.1.0",
    description="MentorEval: A comprehensive benchmark for evaluating student exam grading systems",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Alvaro Francisco Gil",
    author_email="alvaro.francisco.gil@example.com",
    url="https://github.com/alvaro-francisco-gil/mentor-eval",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    install_requires=install_requires,
    dependency_links=git_dependencies,
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "black>=21.0",
            "flake8>=3.8",
            "mypy>=0.800",
        ],
    },
    entry_points={
        "console_scripts": [
            "mentoreval=mentoreval.cli:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Education",
    ],
    keywords="education, grading, evaluation, benchmark, nlp, ai",
)
