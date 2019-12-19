from setuptools import setup, find_packages
from os import path

from io import open

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='biostitch',
    version='0.22',
    description='Image stitcher for Opera Phenix',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/VasylVaskivskyi/biostitch',
    author='Vasyl Vaskivskyi',
    author_email='vaskivskyi.v@gmail.com',
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
    install_requires=['numpy >= 1.17',
                      'pandas >= 0.25',
                      'imagecodecs-lite',
                      'tifffile >= 2019.7.26',
                      'opencv-contrib-python >= 4.0',
                      'dask[complete] >=2.6.0'
                      ]
)
