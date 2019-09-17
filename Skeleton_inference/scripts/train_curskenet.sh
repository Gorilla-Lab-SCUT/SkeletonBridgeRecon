#!/usr/bin/env bash

class='chair'
# train ae
env=ae_cur
python ./training/train_AE_CurSkeNet.py  --num_points 600 --nb_primitives 20 --lr 1e-3 --nepoch 80 --lr_decay_epoch 60 --env $env --category $class
# fix decoder, train svr
env=svr_cur_fix
python ./training/train_SVR_CurSkeNet.py  --num_points 600 --nb_primitives 20 --lr 1e-3 --nepoch 80 --lr_decay_epoch 60 --env $env --category $class --model_preTrained_AE './log/'$class'_log/ae_cur/network.pth'