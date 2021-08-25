#!/usr/bin/env python

import os
# Always prefer setuptools over distutils
from setuptools import setup, find_packages
import pytorch_cpm as module

fpk = find_packages(where=".")
# fpk += ['pytorch_cpm.experiments', 'pytorch_cpm.hrnetv2_pretrained']

setup(
    name=module.__name__,
    version=module.__version__,
    author=module.__author__,
    packages=fpk,
    long_description=module.__description__,
    include_package_data=True,
    zip_safe=False,
    keywords=module.__keywords__,
    python_requires='>=3.6',
    setup_requires=[],
    # install_requires=load_requirements(PATH_ROOT),

    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    entry_points={  
        'console_scripts': [
            'pytorch_cpm= pytorch_cpm.__main__:main'
        ]
    },
    package_data = {
        'pytorch_cpm.runner': ['*.yaml']
    }
)
