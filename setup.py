#!/usr/bin/env python

from setuptools import find_packages, setup

setup(
    name='lpl',
    version='1.0',
    description="Pseudolikelihood Scoring with Masked Language Models",
    author='Julian Salazar',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    entry_points = {
        'console_scripts': ['lpl=lpl.cmds:main'],
    },
    # Needed for static type checking
    # https://mypy.readthedocs.io/en/latest/installed_packages.html
    zip_safe=False
)
