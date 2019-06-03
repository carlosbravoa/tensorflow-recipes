# tensorflow-recipes
Just playing with code to show working examples with TF

## my_TF_object_tdetection.py
A demo for object detection using TF models from the model zoo, as a totally standalone script (i.e. not needing any additional library than cv2 and the standard ones for tensorflow)

This is just a working example, not even optimized, but it can get you running tf models without additional installations or set up.
Just install cv2 (pip install opencv-python), tensorflow and numpy.

There is a lot of better code ready to use from the TF repo. This one was created with the official TF tutorial here: https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb

The model zoo: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

Download one of the models. The script will look for the frozen file. Add the labels in txt format (optional) and ready to go

example: 

`python my_TF_object_detection.py --model ssd_mobilenet_v2_coco_2018_03_29`
It will run with the default camera, using the model deascribed (you have to download it first!), without labels. 

If you want it with the labels:
`python my_TF_object_detection.py --model ./ssd_mobilenet_v2_coco_2018_03_29/ --labels ./coco_labels.txt`

** TODO: adjusting threshold, adding the capability to run classification models ** 

More will come... 
