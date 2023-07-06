#!/bin/bash
accelerate launch train.py --task mnli
accelerate launch train.py --task rte