class='chair'

env='Global_Guidance_'$class

python Global_Guidance/train.py --batchSize 16 --env $env --nepoch 40 --lr 1e-3  --lrDecay 0.1 --lrStep 10 --category $class
