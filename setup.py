from setuptools import setup, find_packages
import sys
import os.path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'tfnet'))

setup(name='tfnet',
      version=0.1,
      description='A helper library for training convnets with tensorflow.',
      url='https://github.com/badgeir/tfnet',
      author='Peter Leupi',
      author_email='pleupi123@gmail.com',
      license='MIT',
      packages=[package for package in find_packages()
                if package.startswith('tfnet')]
      )
