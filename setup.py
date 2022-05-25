from setuptools import setup, find_packages

setup(
    name='algolib',
    version='0.1',
    author='Eric Pfahl',
    packages=find_packages(),
    install_requires=['numpy', 'scipy', 'pandas', 'statsmodels'])
