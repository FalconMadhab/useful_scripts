import numpy as np
import os
import glob
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
import zipfile
import time
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
import cv2
from imutils.video import FileVideoStream,FPS,VideoStream
import pandas as pd
# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.12.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.12.*.')

from utils import label_map_util

from utils import visualization_utils as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'door_training2\\frozen_graph\\frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = 'door_data1\\label_map.pbtxt'
# Loading label map
NUM_CLASSES=2
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
category_index = label_map_util.create_category_index(categories)
print('category_index:',category_index)

detection_graph = tf.Graph()
with detection_graph.as_default():
  od_graph_def = tf.GraphDef()
  with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
    serialized_graph = fid.read()
    od_graph_def.ParseFromString(serialized_graph)
    tf.import_graph_def(od_graph_def, name='')

#category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)
def load_image_into_numpy_array(image):
  (im_width, im_height) = image.size
  return np.array(image.getdata()).reshape(
      (im_height, im_width, 3)).astype(np.uint8)

# PATH_TO_TEST_IMAGES_DIR = 'C:\\Users\\UKS\\wowexpai\\ATM\tensorflow\\models\\research\\object_detection\\door_data\\test_imgs'
#TEST_IMAGE_PATHS = glob.glob(os.path.join(PATH_TO_TEST_IMAGES_DIR, '*.jpg'))

# Size, in inches, of the output images.
IMAGE_SIZE = (12, 8)
'''
def detect_objects(image_as_array,sess,detection_graph):
  #Since the model expect shape as [1, None, None, 3],so we have to expand dimension of the image
  height,width=image_as_array.shape[:2]
  image_expanded=np.expand_dims(image_as_array,axis=0)
  image_tensor=detection_graph.get_tensor_by_name('image_tensor:0')
  
  #Each box represent the detected object
  boxes=detection_graph.get_tensor_by_name('detection_boxes:0')
  
  #Scores corresponding to each box represent the confidence level 
  scores=detection_graph.get_tensor_by_name('detection_scores:0')
  classes = detection_graph.get_tensor_by_name('detection_classes:0')
  num_detections = detection_graph.get_tensor_by_name('num_detections:0')
  
  #Detection session goes here
  (boxes,scores,classes,num_detections)=sess.run([boxes,scores,classes,num_detections],feed_dict={image_tensor: image_expanded})
  print('classes:',classes)
  print('scores:',scores)
  print('boxes:',boxes)
  ymin = int((boxes[0][0][0]*height))
  xmin = int((boxes[0][0][1]*width))
  ymax = int((boxes[0][0][2]*height))
  xmax = int((boxes[0][0][3]*width))
  cv2.rectangle(image_as_array,(xmin,ymin),(xmax,ymax),(0,255,0),2)
  #Visualisation of object detection result
  vis_util.visualize_boxes_and_labels_on_image_array(
        image_as_array,
        np.squeeze(boxes),
        np.squeeze(classes).astype(np.int32),
        np.squeeze(scores),
        category_index,
        use_normalized_coordinates=True,
        line_thickness=8)

  return image_as_array
'''
'''
test_data=pd.read_csv('C:\\Users\\WoWExp\\ujjawal\\ATM\\tensorflow\\helmet_data\\test.csv')
filenames=test_data['filename'].tolist()
labels=test_data['class'].tolist()
pred_labels=[]
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		for file1 in filenames:
			image_path=os.path.join(PATH_TO_TEST_IMAGES_DIR,file1)
			name,ext=file1.split('.')
			print(name)
			image = Image.open(image_path)
			# the array based representation of the image will be used later in order to prepare the
			# result image with boxes and labels on it.
			image_np = load_image_into_numpy_array(image)
			# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
			image_np_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			(boxes,scores,classes,num_detections)=sess.run([boxes,scores,classes,num_detections],feed_dict={image_tensor: image_np_expanded})
			#pred_labels.append(classes)
			vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
			result=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
			cv2.imwrite('C:\\Users\\WoWExp\\ujjawal\\ATM\\tensorflow\\helmet_data\\test_result\\{}.jpg'.format(name),result)
			threshold=0.5
			print(classes)
			for index,value in enumerate(classes[0]):
				if scores[0, index] > threshold:
					pred_labels.append((category_index.get(value)).get('name').encode('utf8'))
print(pred_labels)
print(len(labels))

'''
video_path='door_data1\\14-36-51.flv'
#cap=FileVideoStream(0).start()
cap=cv2.VideoCapture(video_path) 
time.sleep(2)
video_FourCC=cv2.VideoWriter_fourcc(*'DIVX')
video_fps       = cap.get(cv2.CAP_PROP_FPS)
cap.set(5,6)
video_size      = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
cap.set(3,640)
cap.set(4,480)
vid_file=os.path.split(video_path)[-1]
name=vid_file.split('.')
output_path='out.avi'
count=0
out = cv2.VideoWriter(output_path,video_FourCC, video_fps, video_size)
with detection_graph.as_default():
	with tf.Session(graph=detection_graph) as sess:
		while True:
			ret,image=cap.read()
			#result = detect_objects(image,sess, detection_graph)
			if ret == False:
				break
			image_np=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
			image_np=cv2.resize(image_np,(300,300),interpolation=cv2.INTER_AREA)
			image_expanded = np.expand_dims(image_np, axis=0)
			image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
			boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
			scores = detection_graph.get_tensor_by_name('detection_scores:0')
			classes = detection_graph.get_tensor_by_name('detection_classes:0')
			num_detections = detection_graph.get_tensor_by_name('num_detections:0')
			(boxes,scores,classes,num_detections)=sess.run([boxes,scores,classes,num_detections],feed_dict={image_tensor: image_expanded})
			# print('squeeze boxes:',np.squeeze(boxes))
			print('squeeze classes:',np.squeeze(classes).astype(np.int32))
			# print('squeeze scores:',np.squeeze(scores))
			# print('category_index:',category_index)
			vis_util.visualize_boxes_and_labels_on_image_array(image_np,np.squeeze(boxes),np.squeeze(classes).astype(np.int32),np.squeeze(scores),category_index,use_normalized_coordinates=True,line_thickness=8)
			result=cv2.cvtColor(image_np,cv2.COLOR_RGB2BGR)
			cv2.imshow('result',result)
			count=count+1
			cv2.imwrite('door_data1\\test_imgs\\test_vid_5\\test_image'+str(count)+'.jpg', result)
			key = cv2.waitKey(1) & 0xFF

			# if the `q` key was pressed, break from the loop
			if key == ord("q"):
				break
			out.write(result)
cv2.destroyAllWindows()
#cap.stop()
cap.release()
