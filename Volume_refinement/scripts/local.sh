class='chair'

env='Local_Synthesis_'$class

python Local_Synthesis/train.py --batchSize 16 --env $env --nepoch 5 --lr 1e-4 --lrDecay 0.1 --lrStep 2 --category $class --iters 10000
