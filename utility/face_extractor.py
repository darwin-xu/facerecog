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


def cutBoxFromImage(box, image, enlarge=2):
    """ Cut a box from a image with enlarge ratio.
        if the enlarged box is out of the bound of the image.
        random data will be filled.
    """
    imageSize = image.shape[0:2]

    # Get the width and height of the box
    w = box[2] - box[0]
    h = box[3] - box[1]

    # Calculate the enlarged bound of the box, maybe cross the border of the image
    x1 = int(box[0] - w / 2)
    x2 = int(box[2] + w / 2)
    y1 = int(box[1] - h / 2)
    y2 = int(box[3] + h / 2)

    # Initial a random array
    img = np.random.rand(y2 - y1, x2 - x1, image.shape[2])

    # Calculate the inner margin in the enlarge box
    left = 0 if (0 - x1 >= 0) else (0 - x1)
    right = (x2 - x1) if (imageSize[1] - x2 >= 0) else (y2 - y1 -
                                                        (imageSize[1] - x2))
    top = 0 if (0 - y1 >= 0) else (0 - y1)
    bottom = (y2 - y1) if (imageSize[0] - y2 >= 0) else (y2 - y1 -
                                                         (imageSize[0] - y2))

    # Adjust the clipping area
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, imageSize[1])
    y2 = min(y2, imageSize[0])

    img[top:bottom, left:right, :] = image[y1:y2, x1:y2, :]
    return img


def extractFace(imageName, params):
    # Create the folder to save the face images.
    path, filename = getDestPathAndFilename(imageName, 'labeled_faces')
    if not os.path.exists(path):
        os.makedirs(path)
    try:
        image = imageio.imread(imageName)
        bounding_boxes, _ = detect_face.detect_face(
            image, params[0], params[3], params[4], params[5], params[1],
            params[2])
        i = 0
        for box in bounding_boxes:
            face = cutBoxFromImage(box, image)
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
