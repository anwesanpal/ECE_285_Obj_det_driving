# ece285-final-project
Final Project - ECE 285
### Team Name: The Object Detectives

### Description

This is project **Object Detection in the Autonomous Driving Scenario** developed by team *The Object Detectors* composed of Anwesan Pal, David Paz-Ruiz and Harshini Rajachander.

Object Detection in the Autonomous Driving Scenario is a project intended to explore state of the art architectures for real-time multi-object detection for autonomous vehicle detection modules and then attempt to classify the cars into further sub-categories decided by their respective make.

### Requirements
Install packages:
$ `pip install --user -r requirements.txt`

### Code organization
- [Github Repository](https://github.com/AftermathK/ece285-final-project)
- Drive Link: Sent via email.

1. Our Entire Pipeline (DEMO):
    1. git clone https://github.com/AftermathK/ece285-final-project.git
    2. cd ece285-final-project
    2. Download weights for YOLO from this [link](https://pjreddie.com/media/files/yolov3.weights) and save it in the current directory.
    3. Download the model weights for the Resnet18 classifier from the drive link.
    3. Download the images for Vehicle classifier from the Google drive link and place it in the repositary's main directory (Extract `images.zip`)
    4. Run demo_final.ipynb

2. Faster R-CNN: (demo file)
    1. cd Faster R-CNN
    2. Follow instructions given in readme file.
    3. faster_rcc_demo.ipynb - Run main demo file for viewing performance of faster rcnn on single image/video files.

3. RetinaNet
    1. Follow instructions in the given readme.

4. YOLO3
    1. Run the yolo3.ipynb Jupyter notebook in the `yolo3` directory,  This Jupyter notebook assumes that the weights file (yolov3.weights) has been downloaded with the `yolo3` directory and that the sample videos from our dataset have been downloading into the `yolo3` directory.
    2. Download the videos `camera_1.mp4, camera_2.mp4, camera_5.mp4 and camera_6.mp4` from the team drive. These videos are located under Camera-YOLO3/videos (the total size is approximately 2.4GB).
    3. Download the weights provided by Darknet: [link](https://pjreddie.com/media/files/yolov3.weights)
    4. Run the Jupyter notebook's first two cells. Inference time for YOLO on all four of the videos is about 25 minutes on a Titan Xp GPU. The results will be saved in separate directories on the current directory. The results obtained from this detector are also available on the team drive under `YOLO3/ClassificationResults`


5. Experiments folder:
    1. project_train_(Resnet18).ipynb : Run the training for our Vehicle Classifier.
    2. project_train_cars_demo.ipynb  : Run training on experimental networks of VGG16 and Resnet18.
    3. project_train_cars_demo-(AlexVgg19).ipynb : Run training on experimental networks of VGG19 and AlexNet.
