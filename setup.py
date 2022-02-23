import os
from setuptools import setup
from setuptools import find_packages

setup(
    name="distracting_dmc2gym",
    version="1.0.0",
    author="Denis Yarats, Claas Voelcker",
    description=("a gym like wrapper for dm_control and distracting control"),
    license="",
    keywords="gym dm_control openai deepmind",
    packages=find_packages(),
    install_requires=[
        "gym",
        "dm_control",
    ],
)
