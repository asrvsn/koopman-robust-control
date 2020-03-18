#!/bin/bash
echo "Running tests"
python -m sampler.utils
python -m sampler.kernel
python -m sampler.hmc