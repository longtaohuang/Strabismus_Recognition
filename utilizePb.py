#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May  6 10:35:04 2018

@author: JieweiLu
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inception_preprocessing
import tensorflow as tf
import os
import numpy as np

# load the graph
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
    # Initialize the varibales
    input_height = 299
    input_width = 299
    input_layer = "input"
    output_layer = "InceptionV3/Predictions/Reshape_1"
	 # The path of your model file
    model_file = 'C:\\Users\\Taotaozuishuai\\Desktop\\UtilizePb\\frozen_inception_v3.pb'
    # The path of your test images    
    pathOri = 'C:\\Users\\Taotaozuishuai\\Desktop\\UtilizePb\\dataset\\test\\'
    
    # Load the pb model
    graph = load_graph(model_file)
    input_name = "import/" + input_layer
    output_name = "import/" + output_layer
    input_operation = graph.get_operation_by_name(input_name)
    output_operation = graph.get_operation_by_name(output_name)
    
    # Initialize the photo file names
    photo_filenames = []
    filenameAll = os.listdir(pathOri)
    for filename in filenameAll:
        path = os.path.join(pathOri, filename)
        photo_filenames.append(path)
    num_image = len(filenameAll)
    
    actualLabel = []
    predictionLabel = []
    scoreAll = []
    
    with tf.Session(graph=graph) as sess:
        correctNum = 0
		  # Reading images cyclically
        for i in range(num_image):        
            file_reader = tf.gfile.FastGFile(photo_filenames[i], 'rb').read()
            image1 = tf.image.decode_jpeg(file_reader)
            image = inception_preprocessing.preprocess_for_eval(image1,input_height,input_width)
            image = tf.expand_dims(image, 0)
            
            t = sess.run(image)
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})
            results = np.squeeze(results)
			   # [::-1] means descend sorted
            top_2 = results.argsort()[-2:][::-1] 
            score_k = results[top_2]
			   # The probability of high probability
            score_1 = score_k[1] 
            print(results[1])
			   # The classification results, 1 for strabismus, 0 for normal
            top_1 = top_2[0]
            print(top_1)
