import torch
import dist_chamfer as ext

distChamfer =  ext.chamferDist()

i = 0
while True:
    i = i + 1
    x = torch.rand(32, 2500, 3)
    y = torch.rand(32, 2500, 3)
    print(x.size(), y.size())
    x = x.cuda()
    y = y.cuda()
    dis1, dis2 = distChamfer(x, y)
    print(i, dis1.size(), dis2.size())
