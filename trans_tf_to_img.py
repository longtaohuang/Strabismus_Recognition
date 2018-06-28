#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 08:28:22 2018

@author: JieweiLu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from PIL import Image

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import funcRead

def load_graph(model_file):
  graph = tf.Graph()
  graph_def = tf.GraphDef()

  with open(model_file, "rb") as f:
    graph_def.ParseFromString(f.read())
  with graph.as_default():
    tf.import_graph_def(graph_def)

  return graph

if __name__ == "__main__":
    print('Start')
    ## Initialize the varibales
    input_height = 299
    input_width = 299
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"
    model_file = '/home/JieweiLu/jackie/Program/strabisumsProgram/Program1/frozen_inception_v3.pb' 
    pathSave = '/home/JieweiLu/jackie/Program/strabisumsProgram/Program1/saveDataset/test/'
    
    ## load the pb model
    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    
    ## read the slim test dataset
    tf_file = '/home/JieweiLu/jackie/Program/strabisumsProgram/Program1/mydata_validation_00000-of-00001.tfrecord'
    tensor_dict = funcRead.read_TF_slim(tf_file)
    
    
    # define the thread and open the session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord) 
    
    correctNum = 0
    labelAll = []
    for i in range(324):
        print(i)
        
        # extract the information
        diction = sess.run(tensor_dict)
        imageNa = diction[0]
        imageObtain = Image.fromarray(imageNa.astype(np.uint8))
        label = diction[1]       
        labelAll.append(label)
        xu = '%d_%d_IMG.jpg' % (i+1,label)
        imageObtain.save(pathSave+xu)
        
    
    # stop the threads
    coord.request_stop()
    coord.join(threads)
    print('OK')
        
        
    
    