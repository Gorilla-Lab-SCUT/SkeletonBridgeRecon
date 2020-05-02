### Data

* You can prepare the ground truth point cloud (with normals) for learning mesh refinement.
```shell
python ./data/dumpgt.py 
```
* You need to convert ```*.obj``` to  ```.pkl``` to read the vertices, edges, and faces of base meshes.
```shell
python ./data/dump_meshdata.py
```

### Training
* You can train mesh refinement network by run the script: 
```shell
bash scripts/train.sh
```

### Visualizing
* You can run some results for testset, and save them for visualization: 
```shell
bash scripts/save.sh
```

### Evaluation
* You evalute the quantative results like CD, EMD, F-score: 
```shell
bash scripts/eval.sh
```

### Trained models

* We have provided the [trained models](https://drive.google.com/open?id=1occw5YlFUv5cFNH7uQPRmaSg2_Bn1yLe) for testing. 
* You can download the trained models, unzip them, and then go in ```your path/Mesh_refinement/trained_models```
