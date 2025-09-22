"""
Setup configuration for Vortex-Omega NFCS
"""

from setuptools import setup, find_packages
import os

# Чтение README для long_description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Чтение requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Версия
VERSION = "2.4.3"

setup(
    name="vortex-omega-nfcs",
    version=VERSION,
    author="Team Omega",
    author_email="team@vortex-omega.ai",
    description="Neural Field Control System (NFCS) - Advanced AI orchestration framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dukeru115/Vortex-Omega",
    project_urls={
        "Bug Tracker": "https://github.com/dukeru115/Vortex-Omega/issues",
        "Documentation": "https://vortex-omega.readthedocs.io",
        "Source Code": "https://github.com/dukeru115/Vortex-Omega",
    },
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0",
            "pytest-asyncio>=0.21",
            "black>=23.0",
            "flake8>=6.0",
            "mypy>=1.0",
            "pre-commit>=3.0",
        ],
        "docs": [
            "sphinx>=6.0",
            "sphinx-rtd-theme>=1.0",
            "myst-parser>=1.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "vortex-omega=src.main:main",
            "nfcs=src.main:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yml", "*.yaml", "*.json", "*.md"],
    },
    zip_safe=False,
)