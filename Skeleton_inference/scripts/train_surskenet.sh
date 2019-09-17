#!/usr/bin/env bash

class='chair'
# train ae
env=ae_sur
python ./training/train_AE_SurSkeNet.py  --num_points 2000 --nb_primitives 20 --lr 1e-3 --nepoch 80 --lr_decay_epoch 60 --env $env --category $class
# fix decoder, train svr
env=svr_sur_fix
python ./training/train_SVR_SurSkeNet.py  --num_points 2000 --nb_primitives 20 --lr 1e-3 --nepoch 80 --lr_decay_epoch 60 --env $env --category $class --model_preTrained_AE './log/'$class'_log/ae_sur/network.pth'