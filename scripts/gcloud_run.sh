#!/bin/bash
# set -e
gcloud compute instances start pytorch-2-vm --zone=us-west1-a
gcloud compute ssh --ssh-flag="-Y" pytorch-2-vm --zone=us-west1-a
# gcloud compute ssh --ssh-flag="-Y -v" pytorch-2-vm --zone=us-west1-a # If debug
gcloud compute instances stop pytorch-2-vm --zone=us-west1-a