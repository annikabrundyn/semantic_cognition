#!/usr/bin/env python

from setuptools import setup, find_packages

setup(
    name='semantic_cognition',
    version='0.0.0',
    description='Extension of mccullen & rogers model',
    author='Annika Brundyn',
    author_email='ab8690@nyu.edu',
    # REPLACE WITH YOUR OWN GITHUB PROJECT LINK
    url='https://github.com/annikabrundyn/semantic_cognition',
    package_dir={"": "src"},
    packages=find_packages("src"),
)