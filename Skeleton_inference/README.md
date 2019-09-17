### Installation

```shell

cd Skeleton_inference
mkdir log
## Create python env with relevant packages
conda create --name skeleton python=2.7
source activate skeleton
pip install pillow pandas scipy visdom opencv-python plyfile trimesh
conda install pytorch torchvision -c pytorch
```
Tested on pytorch >= 1.0 and python2.7.

### Setting

* Set up enviroments :

```bash
python -m visdom.server -p 9000
export CUDA_VISIBLE_DEVICES=0 #whichever you want
source activate skeleton
```

### Data

We used the rendered imaged from [3d-R2N2](https://github.com/chrischoy/3D-R2N2), and the 3D models from [ShapeNet](https://www.shapenet.org/).
* [The rendered images](https://cloud.enpc.fr/s/S6TCx1QJzviNHq0) go in ```data/ShapeNetRendering```
* [The Curve/Sheet Skeleton(*.ply) datasets](https://drive.google.com/open?id=1cxQmPTYXpATAe4abdE9WojSFPMPMTIaT) go in ```data/ShapeNetPointCloud```
* you can run ```cd data; python ply2mat.py; cd ..``` to convet the *.ply file to *.mat file.

### Build chamfer distance

```shell
source activate skeleton
cd ./extension
python setup.py install
```

### Training

* train the CurSkeNet :
```shell
bash ./scripts/train_curskenet.sh
```

* train the SurSkeNet :
```shell
bash ./scripts/train_surskenet.sh
```

### Demo

```shell
bash ./scripts/demo.sh
```

Next, we will release the code and dataset of patch-based high resolution skeletal volume refinement!
