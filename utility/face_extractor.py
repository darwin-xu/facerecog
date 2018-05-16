# This is a tool to extractor faces from the pictures.
# After save the html, a bunch of files contain faces saved in the folder.
# For instance: <download/images/liudehua>
# This tool will extract the faces and saved into another folder
# For instance: <download/labeled_faces/liudehua>

import os
import sys
import unittest

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


def enlargeBox(box, imageSize, enlarge):
    # Get the width and height of the box
    hExt = (box[2] - box[0]) * enlarge
    vExt = (box[3] - box[1]) * enlarge

    # Calculate the enlarged bound of the box, maybe cross the border of the image
    x1 = int(box[0] - hExt / 2)
    x2 = int(box[2] + hExt / 2)
    y1 = int(box[1] - vExt / 2)
    y2 = int(box[3] + vExt / 2)
    print('x1,x2,y1,y2', x1, x2, y1, y2)

    # Calculate the inner margin in the enlarge box
    left = 0 if (0 <= x1) else (0 - x1)
    right = (x2 - x1) if (x2 <=
                          imageSize['width']) else (imageSize['width'] - x1)
    top = 0 if (0 <= y1) else (0 - y1)
    bottom = (y2 - y1) if (y2 <=
                           imageSize['height']) else (imageSize['height'] - y1)

    print('left, right, top, bottom', left, right, top, bottom)

    # Adjust the clipping area
    x1 = max(x1, 0)
    y1 = max(y1, 0)
    x2 = min(x2, imageSize['width'])
    y2 = min(y2, imageSize['height'])

    return {
        'left': left,
        'right': right,
        'top': top,
        'bottom': bottom,
        'x1': x1,
        'y1': y1,
        'x2': x2,
        'y2': y2
    }


def cutBoxFromImage(box, image, enlarge=2):
    """ Cut a box from a image with enlarge ratio.
        if the enlarged box is out of the bound of the image.
        random data will be filled.
    """

    params = enlargeBox(box, {
        'width': image.shape[0:2][1],
        'height': image.shape[0:2][1]
    }, enlarge)

    # Get the result out of the params
    left, top, right, bottom = params['left'], params['top'], params[
        'right'], params['bottom']
    x1, y1, x2, y2 = params['x1'], params['y1'], params['x2'], params['y2']

    # Initial a random array
    img = np.random.rand(y2 - y1, x2 - x1, image.shape[2])

    # Cut the face out of the image
    img[top:bottom, left:right, :] = image[y1:y2, x1:x2, :]

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


if __name__ == '__main__':
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
            pnet, rnet, onet = detect_face.create_mtcnn(
                sess, '../facenet/src/align')
            minsize = 20
            threshold = [0.6, 0.7, 0.7]  # three steps's threshold
            factor = 0.709  # scale factor
            params = (minsize, threshold, factor, pnet, rnet, onet)

            # For each file, extract the faces
            for imageName in filenames:
                extractFace(imageName, params)


class TestFaceExtractor(unittest.TestCase):
    def testEnlargeBox(self):
        box = [100, 100, 150, 150]
        imageSize = {'width': 250, 'height': 250}
        params = enlargeBox(box, imageSize, enlarge=2)
        self.assertEqual(params['left'], 0)
        self.assertEqual(params['top'], 0)
        self.assertEqual(params['right'], 150)
        self.assertEqual(params['bottom'], 150)
        self.assertEqual(params['x1'], 50)
        self.assertEqual(params['y1'], 50)
        self.assertEqual(params['x2'], 200)
        self.assertEqual(params['y2'], 200)

        box = [140, 210, 190, 290]
        imageSize = {'width': 200, 'height': 300}
        params = enlargeBox(box, imageSize, enlarge=2)
        self.assertEqual(params['left'], 0)
        self.assertEqual(params['top'], 0)
        self.assertEqual(params['right'], 110)
        self.assertEqual(params['bottom'], 170)
        self.assertEqual(params['x1'], 90)
        self.assertEqual(params['y1'], 130)
        self.assertEqual(params['x2'], 200)
        self.assertEqual(params['y2'], 300)

        box = [20, 20, 100, 50]
        imageSize = {'width': 200, 'height': 300}
        params = enlargeBox(box, imageSize, enlarge=2)
        self.assertEqual(params['left'], 60)
        self.assertEqual(params['top'], 10)
        self.assertEqual(params['right'], 240)
        self.assertEqual(params['bottom'], 90)
        self.assertEqual(params['x1'], 0)
        self.assertEqual(params['y1'], 0)
        self.assertEqual(params['x2'], 180)
        self.assertEqual(params['y2'], 80)

        box = [20, 20, 100, 50]
        imageSize = {'width': 200, 'height': 300}
        params = enlargeBox(box, imageSize, enlarge=2)
        self.assertEqual(params['left'], 60)
        self.assertEqual(params['top'], 10)
        self.assertEqual(params['right'], 240)
        self.assertEqual(params['bottom'], 90)
        self.assertEqual(params['x1'], 0)
        self.assertEqual(params['y1'], 0)
        self.assertEqual(params['x2'], 180)
        self.assertEqual(params['y2'], 80)
