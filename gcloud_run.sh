#!/bin/bash
# set -e
gcloud compute instances start pytorch-1-vm --zone=us-west1-b
gcloud compute ssh pytorch-1-vm --zone=us-west1-b
gcloud compute instances stop pytorch-1-vm --zone=us-west1-b