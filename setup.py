#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Setup uSFGAN library."""

import os
import pip
import sys

from distutils.version import LooseVersion
from setuptools import find_packages
from setuptools import setup

if LooseVersion(sys.version) < LooseVersion("3.8"):
    raise RuntimeError(
        "usfgan requires Python>=3.8, "
        "but your Python is {}".format(sys.version))
if LooseVersion(pip.__version__) < LooseVersion("20.0.0"):
    raise RuntimeError(
        "pip>=20.0.0 is required, but your pip is {}. "
        "Try again after \"pip install -U pip\"".format(pip.__version__))

requirements = {
    "install": [
        "torch>=1.7.1",
        "setuptools>=38.5.1",
        "librosa>=0.8.0",
        "soundfile>=0.10.2",
        "tensorboardX>=1.8",
        "matplotlib>=3.1.0",
        "PyYAML>=3.12",
        "tqdm>=4.26.1",
        "kaldiio>=2.14.1",
        "h5py>=2.10.0",
        "pyworld>=0.2.12",
        "docopt",
        "sprocket-vc",
    ],
    "setup": [
        "numpy",
        "pytest-runner",
    ],
    "test": [
        "pytest>=3.3.0",
        "hacking>=1.1.0",
        "flake8>=3.7.8",
        "flake8-docstrings>=1.3.1",
    ]
}
entry_points = {
    "console_scripts": [
        "usfgan-preprocess=usfgan.bin.preprocess:main",
        "usfgan-compute-statistics=usfgan.bin.compute_statistics:main",
        "usfgan-train=usfgan.bin.train:main",
        "usfgan-decode=usfgan.bin.decode:main",
    ]
}

install_requires = requirements["install"]
setup_requires = requirements["setup"]
tests_require = requirements["test"]
extras_require = {k: v for k, v in requirements.items()
                  if k not in ["install", "setup"]}

dirname = os.path.dirname(__file__)
setup(name="usfgan",
      version="0.1",
      url="http://github.com/chomeyama/UnifiedSourceFilterGAN",
      author="Reo Yoneyama",
      author_email="yoneyama.reo@g.sp.m.is.nagoya-u.ac.jp",
      description="Unified Source-Filter GAN implementation",
      long_description_content_type="text/markdown",
      long_description=open(os.path.join(dirname, "README.md"), encoding="utf-8").read(),
      license="MIT License",
      packages=find_packages(include=["usfgan*"]),
      install_requires=install_requires,
      setup_requires=setup_requires,
      tests_require=tests_require,
      extras_require=extras_require,
      entry_points=entry_points,
      classifiers=[
          "Programming Language :: Python :: 3.8.6",
          "Intended Audience :: Science/Research",
          "Operating System :: POSIX :: Linux",
          "License :: OSI Approved :: MIT License",
          "Topic :: Software Development :: Libraries :: Python Modules"],
      )
