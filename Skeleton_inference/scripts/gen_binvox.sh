#!/usr/bin/env bash

class='table'

# 32 train 
python ./generation/run_SVR_SkeletonRecon_gen_binvox.py --model_line  './trained_models/'$class'/svr_cur/network.pth' --model_square './trained_models/'$class'/svr_sur/network.pth' --gen_line_points 2400 --gen_square_points 32000 --nb_primitives_line 20 --nb_primitives_square 20 --trainset --res 32 --category $class

# 32 test
python ./generation/run_SVR_SkeletonRecon_gen_binvox.py --model_line  './trained_models/'$class'/svr_cur/network.pth' --model_square './trained_models/'$class'/svr_sur/network.pth' --gen_line_points 2400 --gen_square_points 32000 --nb_primitives_line 20 --nb_primitives_square 20 --res 32 --category $class

# 64 train 
python ./generation/run_SVR_SkeletonRecon_gen_binvox.py --model_line  './trained_models/'$class'/svr_cur/network.pth' --model_square './trained_models/'$class'/svr_sur/network.pth' --gen_line_points 2400 --gen_square_points 32000 --nb_primitives_line 20 --nb_primitives_square 20 --trainset --res 64 --category $class

# 64 test
python ./generation/run_SVR_SkeletonRecon_gen_binvox.py --model_line  './trained_models/'$class'/svr_cur/network.pth' --model_square './trained_models/'$class'/svr_sur/network.pth' --gen_line_points 2400 --gen_square_points 32000 --nb_primitives_line 20 --nb_primitives_square 20 --res 64 --category $class

# 128 train 
python ./generation/run_SVR_SkeletonRecon_gen_binvox.py --model_line  './trained_models/'$class'/svr_cur/network.pth' --model_square './trained_models/'$class'/svr_sur/network.pth' --gen_line_points 2400 --gen_square_points 32000 --nb_primitives_line 20 --nb_primitives_square 20 --trainset --res 128 --category $class 

# 128 test
python ./generation/run_SVR_SkeletonRecon_gen_binvox.py --model_line  './trained_models/'$class'/svr_cur/network.pth' --model_square './trained_models/'$class'/svr_sur/network.pth' --gen_line_points 2400 --gen_square_points 32000 --nb_primitives_line 20 --nb_primitives_square 20 --res 128 --category $class
