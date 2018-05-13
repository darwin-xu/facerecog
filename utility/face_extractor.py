# This is a tool to extractor faces from the pictures.
# After save the html, a bunch of files contain faces saved in the folder.
# For instance: <download/images/liudehua>
# This tool will extract the faces and saved into another folder
# For instance: <download/labeled_faces/liudehua>

import os
import sys

import imageio
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from PIL import ImageEnhance
from scipy import misc, ndimage

import detect_face


def getDestPathAndFilename(filename, destName):
    folder = os.path.dirname(filename)
    individual = os.path.basename(folder)
    download = os.path.dirname(os.path.dirname(folder))
    dest = os.path.join(download, destName, individual)
    name = os.path.splitext(os.path.basename(filename))[0]
    return (dest, name)


def extractFace(imageName, params):
    # Create the folder to save the face images.
    path, filename = getDestPathAndFilename(imageName, 'labeled_faces')
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        image = imageio.imread(imageName)
        imageSize = image.shape[0:2]
        bounding_boxes, _ = detect_face.detect_face(
            image, params[0], params[3], params[4], params[5], params[1],
            params[2])
        i = 0
        for box in bounding_boxes:
            x1 = int(max(box[0], 0))
            y1 = int(max(box[1], 0))
            x2 = int(min(box[2], imageSize[1]))
            y2 = int(min(box[3], imageSize[0]))
            face = image[y1:y2, x1:x2, :]
            outImageName = os.path.join(path, filename + '_' + str(i) + '.png')
            i += 1
            print('Save to' + outImageName)
            imageio.imsave(outImageName, face)
    except ValueError:
        pass


# Get the filenames in the folder
folder = os.path.abspath(sys.argv[1])
filenames = [
    os.path.join(folder, dir) for dir in os.listdir(folder)
    if os.path.isfile(os.path.join(folder, dir))
]

# Create graph and sess for tensorflow
graph = tf.get_default_graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,
                                                    '../facenet/src/align')
        minsize = 20
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        params = (minsize, threshold, factor, pnet, rnet, onet)

        # For each file, extract the faces
        for imageName in filenames:
            extractFace(imageName, params)
