import os

from setuptools import find_packages
from setuptools import setup

CWD = os.path.abspath(os.path.dirname(__file__))

if "VERSION" in os.environ:
    version = os.environ["VERSION"]
else:
    version = "0.1"

setup(
    name="pycss",
    description="css exports to Python.",
    author="Anav Sood",
    author_email="anavsood@stanford.edu",
    packages=find_packages(),
    install_requires=["numpy", "pybind11"],
    data_files=[("../../pycss", ["pycss_core.so"])],
    zip_safe=False,
    version=version,
)