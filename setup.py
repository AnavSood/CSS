import os

from setuptools import find_packages
from setuptools import setup

CWD = os.path.abspath(os.path.dirname(__file__))

if "VERSION" in os.environ:
    version = os.environ["VERSION"]
else:
    version = "0.0.9"

setup(
    name="pycss",
    description="CSS Package",
    author="Anav Sood",
    author_email="anavsood@stanford.edu",
    packages=find_packages(),
    install_requires=["numpy"],
    zip_safe=False,
    version=version,
)