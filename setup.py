from setuptools import find_packages

from setuptools import find_namespace_packages
from setuptools import setup

requirements = ['torch>=1.3.1', "torchvision>=0.4", "tensorboardX>=1.4"]

print(find_namespace_packages())
setup(
    name="monodepth2",
    version="0.0.1",
    packages=find_packages(),
    #install_requires=requirements,
    classifiers=("Programming Language :: Python :: 3")
)