#!/bin/bash
BASE_MODEL=$1
accelerate launch train.py --task mnli --base_model $BASE_MODEL
accelerate launch train.py --task rte --base_model $BASE_MODEL