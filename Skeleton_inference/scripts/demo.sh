#!/usr/bin/env bash

class='chair'

python ./generation/run_SVR_SkeletonRecon_demo.py --model_line  './trained_models/svr_line20_pretrained_'$class'.pth' --model_square './trained_models/svr_square20_pretrained_'$class'.pth' --gen_line_points 600 --gen_square_points 2000 --nb_primitives_line 20 --nb_primitives_square 20
