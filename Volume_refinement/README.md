### Data

* You can download the online dataset(including initial and true skeletal volume) for volume refinement learning.
* [The dataset](https://drive.google.com/open?id=1Vc1MNL1EZPndcWs5tP-hehLPQytn0VQA) go in ```your path/Volume_refinement/data```

### Training
* In the released code, we train the global guidance and local synthesis network separately. 

* train the global guidance network:
```shell
bash scripts/global.sh
```

* train the sub-volume synthesis network:
```shell
bash scripts/local.sh
```

### Trained models

* We have provided the [trained models](https://drive.google.com/open?id=15iAaCZGdPIIlJJaXxWMtTkiws0A8NnUX) for testing. 
* You can download the trained models, unzip them, and then go in ```your path/Volume_refinement/trained_models```

### Base mesh generation

* you need to run the script to extract base meshes and simplify them:
```shell
bash scripts/base_mesh.sh
```

* We apply marchinge cubes and conduct mesh simplication, used in this paper [Occupancy Networks - Learning 3D Reconstruction in Function Space](https://github.com/autonomousvision/occupancy_networks/tree/master/external/mesh-fusion). Please also cite it if you use the code.
* We decimate the number of trianlges in base mesh to 10000. (The number is different from the setting in our paper.)
