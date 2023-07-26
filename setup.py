#!/usr/bin/env python

"""The setup script."""

from setuptools import find_packages, setup

with open("README.rst") as readme_file:
    readme = readme_file.read()

with open("HISTORY.rst") as history_file:
    history = history_file.read()

requirements = [
    "bokeh==3.2.0",
    "loguru==0.6.0",
    "numpy==1.25.1",
    "terra-sdk==2.0.6",
    "terra-proto==1.0.1",
]

test_requirements = [
    "pytest>=3",
]

setup(
    author="",
    author_email="",
    python_requires=">=3.10",
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: Apache Software License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
    ],
    description="A package for quantitative analysis and easy data visualisation of constant product automated market makers (CP AMMs)",  # noqa
    install_requires=requirements,
    license="Apache Software License 2.0",
    long_description=readme + "\n\n" + history,
    include_package_data=True,
    keywords="terra_algo_backtest",
    name="terra_algo_backtest",
    packages=find_packages(include=["terra_algo_backtest", "terra_algo_backtest.*"]),
    test_suite="tests",
    tests_require=test_requirements,
    url="https://github.com/terra-revival/terra_algo_backtest",
    version="0.2.0",
    zip_safe=False,
)
