gpu=0
data_list='./data/03001627_trainlist.txt'
epochs=20
dirname='chair'
we=300
wn=2

CUDA_VISIBLE_DEVICES=$gpu python train.py --data_list $data_list --epochs $epochs \
     --checkpoint_dir $dirname --weight_edge $we --weight_norm $wn
