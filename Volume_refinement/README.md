### Data

* You can download the online dataset(including initial and true skeletal volume) for volume refinement learning.
* The [volume_refinement_dataset](https://drive.google.com/drive/folders/19a8rBLl5zt9dv2RVnbROhbudRQV10bZE) go in ```your path/Volume_refinement/data```

### Training
* In the released code, the global guidance and local synthesis network are trained separately. 

* train the global guidance network:
```shell
bash scripts/global.sh
```

* train the sub-volume synthesis network:
```shell
bash scripts/local.sh
```

### Trained models

* We have provided the [volume_trained_models.zip](https://drive.google.com/drive/folders/19a8rBLl5zt9dv2RVnbROhbudRQV10bZE) for testing. 
* You can download the trained models, unzip them, and then go in ```your path/Volume_refinement/trained_models```

### Base mesh generation

* you need to run the script to extract base meshes and simplify them:
```shell
bash scripts/base_mesh.sh
```

* We apply marchinge cubes algorithm and conduct mesh simplication, used in this paper [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://github.com/autonomousvision/occupancy_networks/tree/master/external/mesh-fusion). Please also cite it if you use the code.
* We decimate the number of trianlges in base mesh to 10000. (The number is different from the setting in our paper.)
