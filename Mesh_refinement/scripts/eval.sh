gpu=0
data_list='./data/03001627_testlist.txt'
dirname='chair'

CUDA_VISIBLE_DEVICES=$gpu python eval.py --data_list $data_list --checkpoint_dir $dirname --vertex_chamfer False