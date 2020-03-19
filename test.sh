#!/bin/bash
echo "Running tests"
set -e
python -m sampler.utils
python -m sampler.kernel
python -m sampler.hmc
python -m sampler.hmc_parallel
python -m sampler.dynamics