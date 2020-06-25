# YOLOv3
YOLOv3 algorithm deployed with pytorch

# Inference
To Inference, you first need to:

> git clone https://github.com/Harvey9610/YOLOv3.git
> cd weights
> wget https://pjreddie.com/media/files/yolov3.weights


Then, you can run
> python detect.py

By default, the code will detect the objects in all the picture under the data/samples folder and then save the output picture under data/outputs.

# train
To train, you just run
> python train.py
