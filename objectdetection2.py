import os 
import cv2 
import numpy as np 
import tensorflow as tf 
import sys
#################################################################################
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.98)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

frame_w = frame_h = 512
#conf = tf.ConfigProto()
#conf.gpu_options.allow_growth=True
#session = tf.Session(config=conf)
########################################################################################3
# This is needed since the notebook is stored in the object_detection folder. 
sys.path.append("..") 
  
# Import utilites 
from object_detection.utils import label_map_util 
from object_detection.utils import visualization_utils as vis_util

  
# Name of the directory containing the object detection module we're using 
#MODEL_NAME = 'ssd_mobilenet_v1_coco_11_06_2017' # The path to the directory where frozen_inference_graph is stored. 
IMAGE_NAME = 'north.png'  # The path to the image in which the object has to be detected. !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
  
# Grab path to current working directory 
CWD_PATH = os.getcwd() 
  
# Path to frozen detection graph .pb file, which contains the model that is used 
# for object detection. 
#PATH_TO_CKPT = 'fine_tuned_model/frozen_inference_graph.pb'
PATH_TO_CKPT = 'frozen_inference_graph.pb'
  
# Path to label map file 
PATH_TO_LABELS = 'annotations/label_map.pbtxt'
  
# Path to image 
PATH_TO_IMAGE =  'Ambulances-Stuck-in-Traffic-Jam-in-City-1.jpg'
# Number of classes the object detector can identify 
NUM_CLASSES = 1
  
# Load the label map. 
# Label maps map indices to category names, so that when our convolution 
# network predicts `5`, we know that this corresponds to `king`. 
# Here we use internal utility functions, but anything that returns a 
# dictionary mapping integers to appropriate string labels would be fine 
label_map = label_map_util.load_labelmap(PATH_TO_LABELS) 
categories = label_map_util.convert_label_map_to_categories( 
        label_map, max_num_classes = NUM_CLASSES, use_display_name = True) 
category_index = label_map_util.create_category_index(categories) 
  
# Load the Tensorflow model into memory. 
detection_graph = tf.Graph() 
with detection_graph.as_default():
    with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        od_graph_def = tf.compat.v1.GraphDef()
        od_graph_def.ParseFromString(fid.read())
        tf.import_graph_def(od_graph_def, name ='') 
  
    sess = tf.compat.v1.Session(graph = detection_graph) 
  
# Define input and output tensors (i.e. data) for the object detection classifier 
  
# Input tensor is the image 
image_tensor = detection_graph.get_tensor_by_name('image_tensor:0') 
  
# Output tensors are the detection boxes, scores, and classes 
# Each box represents a part of the image where a particular object was detected 
detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0') 
  
# Each score represents level of confidence for each of the objects. 
# The score is shown on the result image, together with the class label. 
detection_scores = detection_graph.get_tensor_by_name('detection_scores:0') 
detection_classes = detection_graph.get_tensor_by_name('detection_classes:0') 
  
# Number of objects detected 
num_detections = detection_graph.get_tensor_by_name('num_detections:0') 
  
# Load image using OpenCV and 
# expand image dimensions to have shape: [1, None, None, 3] 
# i.e. a single-column array, where each item in the column has the pixel RGB value
#########################################################################################################


#########################################################################################################
image = cv2.imread(PATH_TO_IMAGE)
image_expanded = np.expand_dims(image, axis = 0) 
  
# Perform the actual detection by running the model with the image as input 
(boxes, scores, classes, num) = sess.run( 
    [detection_boxes, detection_scores, detection_classes, num_detections], 
    feed_dict ={image_tensor: image_expanded}) 
  
# Draw the results of the detection (aka 'visualize the results') 
  
# vis_util.visualize_boxes_and_labels_on_image_array(
#     image,
#     np.squeeze(boxes),
#     np.squeeze(classes).astype(np.int32),
#     np.squeeze(scores),
#     category_index,
#     use_normalized_coordinates = True,
#     line_thickness = 2,
#     min_score_thresh = 0.90)

labels = []
with open('./labels.txt', 'r') as file:
	for line in file.read().splitlines():
		a = line.split()#.readline()
		a = a[-1]
		#label = label.replace('\n', '')
		a = str(a)
		labels.append(a)


best_boxes_roi = []
best_boxes_scores = []
best_boxes_classes = []
for i in range(boxes.shape[0]):
    temp = boxes[i, :30] * 512
    best_boxes_roi.append(temp)
    best_boxes_scores.append(scores[i, :30])
    best_boxes_classes.append(classes[i, :30])
best_boxes_roi = np.asarray(best_boxes_roi)
best_boxes_scores = np.asarray(best_boxes_scores)
best_boxes_classes = np.asarray(best_boxes_classes)
classes=best_boxes_classes
#print(labels,classes)
for i in range(best_boxes_roi.shape[0]):
    im = np.reshape(image_expanded[i], (512, 512, 3))

    for j in range(30):

        if best_boxes_scores[i][j] > 0.50 and int(classes[i][j])==8 :
            x = int(best_boxes_roi[i][j][1])
            y = int(best_boxes_roi[i][j][0])
            x_max = int(best_boxes_roi[i][j][3])
            y_max = int(best_boxes_roi[i][j][2])
            crop_img=im[y:y+y_max,x:x+x_max]
            cv2.imwrite('croped_image.png',crop_img)
            cv2.rectangle(im, (x, y), (500, 500), (0, 255, 0), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX

            cv2.putText(im, labels[int(classes[i][j])], (x, y), font, 1e-3 * 512, (255, 0, 0), 2)
        # cv2.imshow('Output',im)
            cv2.imshow('Object detector', im)
            cv2.imwrite('ambu.png', im)
            with open('val.txt', 'w') as file:
                di={PATH_TO_IMAGE:str(best_boxes_scores[i][j])}
                file.write(str(di))

# All the results have been drawn on the image. Now display the image. 
#
# cv2.imshow('Object detector', image)
# cv2.imwrite('ambu.jpg', image)
# Press any key to close the image 
cv2.waitKey(0) 
  
# Clean up 
cv2.destroyAllWindows()
