# CSS

## Installation

1. If you do not have conda installed already, please install it. There are many ways to get conda. We recommend installing Mambaforge which is a conda installation with mamba installed by default and set to use conda-forge as the default set of package repositories. 

2. Clone the git repo:
```
git clone git@github.com:AnavSood/CSS.git
```

3. Set up your conda environment. The list of packages that will be installed inside your conda environment can be seen in `environment.yml`.
```
mamba update -y conda
# create a development virtual environment with useful tools
mamba env create
conda activate CSS
```
