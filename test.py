from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from scipy import misc
import cv2
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib



video_capture = cv2.VideoCapture('./source.mp4')

# #video writer
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('demo.avi', fourcc, fps=30, frameSize=(640,480))

print('Start Recognition!')
prevTime = 0
while True:
    ret, frame = video_capture.read()

    if not ret:
        break

    out.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
# #video writer
out.release()
cv2.destroyAllWindows()
print ('done')
