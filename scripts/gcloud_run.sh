#!/bin/bash
# set -e
gcloud compute instances start pytorch-2-vm --zone=us-west1-a
gcloud compute ssh --ssh-flag="-Y" pytorch-2-vm --zone=us-west1-a
# echo "Running tests"
# set -e
# python -m sampler.utils
# python -m sampler.kernel
# python -m sampler.hmc
# python -m sampler.hmc_parallel
# python -m sampler.ugen
# gcloud compute ssh --ssh-flag="-Y -v" pytorch-2-vm --zone=us-west1-a # If debug
gcloud compute instances stop pytorch-2-vm --zone=us-west1-a