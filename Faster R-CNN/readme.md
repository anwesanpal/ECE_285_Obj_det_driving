# Instructions to run Faster-RCNN for vehicle object detection: 

1. git clone https://github.com/jwyang/faster-rcnn.pytorch.git
2. cd faster-rcnn.pytorch
3. pip install -r requirements.txt
4. cd lib
5. sh make.sh  
6. mkdir data
7. cd data (place test images here) 
8. mkdir pretrained_model (place  models here. Download vgg16 [here](https://www.dropbox.com/s/6ief4w7qzka6083/faster_rcnn_1_6_10021.pth?dl=0) and resnet [here](https://www.dropbox.com/s/5if6l7mqsi4rfk9/faster_rcnn_1_6_14657.pth?dl=0) or [here](https://www.dropbox.com/s/y171ze1sdw1o2ph/faster_rcnn_1_6_9771.pth?dl=0))

8. ipython2 notebook (requires python 2.7 and pytorch==0.4)
9. run Display_final.ipynb for COCO Resnet101 
