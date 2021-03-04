This repository constains the codes and [ShapeNet-Skeleton.zip](https://drive.google.com/drive/folders/19a8rBLl5zt9dv2RVnbROhbudRQV10bZE) datasets for the [Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Tang_A_Skeleton-Bridged_Deep_Learning_Approach_for_Generating_Meshes_of_Complex_CVPR_2019_paper.pdf).

This implementation uses [Pytorch](http://pytorch.org/) and [TensorFlow](https://www.tensorflow.org/).

### Implementation details
For each stage, please follow the README.md under the ```Skeleton_inference/Volume_refinement/Mesh_refinement``` folder.

### Fast demo
* Skeleton inference from the RGB images, and then extract coarse meshes from refined volumes.
```shell
python demo_im2mesh.py
```
* Reuse the input images to deform coarse meshes for surface fitting.
```shell
python demo_deform.py
```

### Citing this work
If you find this work useful in your research, please consider citing:
```shell
@InProceedings{Tang_2019_CVPR,
author = {Tang, Jiapeng and Han, Xiaoguang and Pan, Junyi and Jia, Kui and Tong, Xin},
title = {A Skeleton-Bridged Deep Learning Approach for Generating Meshes of Complex Topologies From Single RGB Images},
booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2019}
}
```
