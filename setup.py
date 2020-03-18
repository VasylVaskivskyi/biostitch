from io import open
from os import path

import sys
import os

from setuptools import setup, find_packages

_base_dir = os.path.dirname(os.path.abspath(__file__))
_requirements = os.path.join(_base_dir, 'requirements.txt')

install_requirements = []
with open(_requirements) as f:
    install_requirements = f.read().splitlines()

long_description = ""
with open(os.path.join(_base_dir, 'README.md'), 'rb') as f:
    long_description = f.read().decode('utf-8')

setup(
    name='biostitch',
    version='0.3.0',
    url='https://github.com/VasylVaskivskyi/biostitch',
    author='Vasyl Vaskivskyi',
    author_email='vaskivskyi.v@gmail.com',
    description='Image stitcher for Opera Phenix',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(exclude=['ez_setup']),
    include_package_data=True,
    zip_safe=False,
    install_requires=install_requirements,
    license='Apache Software License',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
    python_requires='>=3.6, <4',
    entry_points={
        'console_scripts': [
            'biostitch=biostitch.__main__:main',
        ],
    },
)
