#!/bin/bash
accelerate launch train.py --task mnli --epoch 100
# accelerate launch train.py --task rte
# accelerate launch train.py --task mrpc
# accelerate launch train.py --task qnli
# accelerate launch train.py --task qqp
# accelerate launch train.py --task sst2
# accelerate launch train.py --task stsb
# accelerate launch train.py --task wnli

# accelerate launch train.py --task mnli --layer 0,1,2,3,4,5
# accelerate launch train.py --task rte --layer 6,7,8,9,10,11