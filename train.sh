#!/bin/bash
accelerate launch train.py --task mnli --layer 0,1,2,3,4,5
accelerate launch train.py --task rte --layer 6,7,8,9,10,11