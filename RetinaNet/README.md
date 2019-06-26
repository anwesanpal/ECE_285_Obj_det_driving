# Instructions to run RetinaNet for vehicle object detection:

1. git clone https://github.com/yhenon/pytorch-retinanet.git
2. Install the required packages
```
apt-get install tk-dev python-tk
```
3. Install the python packages
```
pip install cffi

pip install pandas

pip install pycocotools

pip install cython

pip install opencv-python

pip install requests

```
4. Build the NMS extension.

```
cd pytorch-retinanet/lib
bash build.sh
cd ../
```

Note that you may have to edit line 14 of `build.sh` if you want to change which version of python you are building the extension for.

## Pre-trained model

A pre-trained model is available at: 
- https://drive.google.com/open?id=1yLmjq3JtXi841yXWBxst0coAgR26MNBS (this is a pytorch state dict)
- https://drive.google.com/open?id=1hCtM35R_t6T8RJVSd74K4gB-A1MR-TxC (this is a pytorch model serialized via `torch.save()`)

The state dict model can be loaded using:

```
retinanet = model.resnet50(num_classes=dataset_train.num_classes(),)
retinanet.load_state_dict(torch.load(PATH_TO_WEIGHTS))
```

The pytorch model can be loaded directly using:

```
retinanet = torch.load(PATH_TO_MODEL)
```
5. Run evaluate.py or the equivalent retinanet.ipynb for detection on a video/single image.
