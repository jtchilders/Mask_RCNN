import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt
import cv2,time,json,glob
#from IPython.display import clear_output


# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
# Import COCO config
sys.path.append(os.path.join("/home/jchilders/coco/"))  # To find local version
from coco import coco
import tensorflow as tf
print('tensorflow version: ',tf.__version__)
print('using gpu: ',tf.test.is_gpu_available())
#%matplotlib inline 

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

# Directory of images to run detection on
IMAGE_DIR = os.path.join(ROOT_DIR, "images")

# COCO Class names
# Index of the class in the list is its ID. For example, to get ID of
# the teddy bear class, use: class_names.index('teddy bear')
class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
               'bus', 'train', 'truck', 'boat', 'traffic light',
               'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
               'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
               'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
               'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
               'kite', 'baseball bat', 'baseball glove', 'skateboard',
               'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
               'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
               'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
               'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
               'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
               'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
               'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
               'teddy bear', 'hair drier', 'toothbrush']

def get_video_data(fn,model,batch_size,show_img=False):
    cap = cv2.VideoCapture(fn)
    print('cap=',cap.isOpened())
    fps = cap.get(cv2.CAP_PROP_FPS)
    print('frames per second: %d' % fps)

    frames = []
    ret, frame = cap.read()
    timestamp = [cap.get(cv2.CAP_PROP_POS_MSEC)]
    frames.append(frame)
    data = []
    output = {'filename:': fn, 
              'fps': fps,
              'timestamp': str(time.ctime(os.path.getmtime(fn))),
              'data': data}
    while(ret):

        if len(frames) == batch_size:
            results = model.detect(frames)
            for i in range(len(results)):
                r = results[i]
                rois = r['rois'].tolist()
                masks = r['masks'] * 1
                class_ids = r['class_ids']
                size = []
                position = []
                pixel_size = []
                class_name = []
                for i in range(len(rois)):
                    size.append([ rois[i][2] - rois[i][0],
                                  rois[i][3] - rois[i][1] ])
                    position.append([ rois[i][0]+int(float(size[-1][0])/2.),
                                      rois[i][1]+int(float(size[-1][1])/2.) ] )
                    pixel_size.append(int(masks[i].sum()))
                    class_name.append(class_names[class_ids[i]])
                data.append({'size': size, 
                             'position': position,
                             'pixel_size': pixel_size,
                             'frametime': timestamp[i],
                             'rois':rois,
                             'class_ids':r['class_ids'].tolist(),
                             'class_names':class_name,
                             'scores':r['scores'].tolist()})
            if show_img:
                clear_output(wait=True)
                vr = results[0]
                visualize.display_instances(frames[0], vr['rois'], vr['masks'], vr['class_ids'], 
                                            class_names, vr['scores'])
#             print(r['rois'])
#             print(r['class_ids'])
#             print(r['scores'])
#             json.dump(data,open('%s_fps%d.json' % (os.path.basename(fn),fps),'w'),indent=2, sort_keys=True)

            frames = []
            timestamp = []
        ret, frame = cap.read()
        timestamp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        frames.append(frame)
    return output

batch_size = 25
class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = batch_size
    BATCH_SIZE = batch_size
config = InferenceConfig()

# Create model object in inference mode.
model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

# Load weights trained on MS-COCO
model.load_weights(COCO_MODEL_PATH, by_name=True)


filelist = open('/home/jchilders/car_videos/filelist.txt').readlines()
print('files: %d' % len(filelist))
output = []
for i,line in enumerate(filelist):
    print(' %s of %s' % (i,len(filelist)))
    fn = line.strip()
    if os.path.exists(fn):    
        fn_output = get_video_data(fn,model,batch_size,show_img=True)
        print(fn_output)
        #clear_output(wait=True)
        output.append(fn_output)
    else:
        print('filename does not exist: ',fn)

json.dump(output,open('full_data.json'))