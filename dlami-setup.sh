#!/usr/bin/env bash

set -x

# We use pipenv for our virtual environment:
pip install --user --upgrade pipenv
export PATH="$PATH:~/.local/bin"
# We require Python 3.7. To install:
sudo apt -y install aptdaemon
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt -y update
# https://askubuntu.com/questions/132059/how-to-make-a-package-manager-wait-if-another-instance-of-apt-is-running/231859#231859
yes | sudo aptdcon --hide-terminal --safe-upgrade
sudo apt -y install bc python3.7 python3.7-dev
# We also require CUDA 10.1 (e.g., there are memory leaks on the default CUDA 9.0). To switch:
sudo rm /usr/local/cuda
sudo ln -s /usr/local/cuda-10.0 /usr/local/cuda
# After this, the included Pipfile should work
# We install this package as editable, with dev packages installed, using Python 3.7
pipenv install --python 3.7 --dev -e .
# Register the virtual environment in ipython
pipenv run ipython kernel install --user --name=lpl-env
# Enter the virtual environment
pipenv shell
