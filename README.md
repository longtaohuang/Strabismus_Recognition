# Strabismus_Recognition
This repository utilizes the Pb models to recognize strabismus.
## Content
* Dataset: The test images tend to be recognized.
* frozen_inception_v3.pb: The Pb models that contain the variables we after trained the networks.
* funcRead.py: The function that read the TFRecord.
* inception_preprocessing.py: The function that preprocess the images to do data augmentation.
* mydata_validation_00000-of-00001.tfrecord: The TFRecord files.
* trans_tf_to_img.py: The function that translate the TFRecord to images.
* utilizePb.py: The main function that utilize the Pb models to predict the results whether it is strabismus.
## Usage
To `run` utilizePb.py after you `change the path` of modle_file and pathOri(`The Pb models and the test iamges`), <br>
then you will get the highest predict probability of strabismus and the result by the model(1 for strabismus and 0 for normal).
## Requirement
Platform: Windows, Ubuntu. <br>
Tensorflow: The version is higher than 1.40.
