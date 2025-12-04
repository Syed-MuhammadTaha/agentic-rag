"""Setup script for agentic-rag package."""

from setuptools import find_packages, setup

setup(
    name="agentic-rag",
    version="0.1.0",
    packages=find_packages(include=["app", "app.*"]),
    python_requires=">=3.9",
)
