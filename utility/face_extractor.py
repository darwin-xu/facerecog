#This is a tool to extractor faces from the picture.

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


def sharpen(image, factor):
    im = PIL.Image.fromarray(image)
    enhancer = ImageEnhance.Sharpness(im)
    return np.asarray(enhancer.enhance(factor))


image = imageio.imread(sys.argv[1])
imageSize = image.shape[0:2]
path = os.path.dirname(sys.argv[1])
fileName = os.path.splitext(os.path.basename(sys.argv[1]))[0]

graph = tf.get_default_graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,
                                                    '../facenet/src/align')

        minsize = 20
        threshold = [0.6, 0.7, 0.7]  # three steps's threshold
        factor = 0.709  # scale factor
        bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet,
                                                    onet, threshold, factor)

        pathForFaces = os.path.join(path, fileName)
        if not os.path.exists(pathForFaces):
            os.makedirs(pathForFaces)

        i = 0
        show = False
        f = 4
        for box in bounding_boxes:
            x1 = int(max(box[0], 0))
            y1 = int(max(box[1], 0))
            x2 = int(min(box[2], imageSize[1]))
            y2 = int(min(box[3], imageSize[0]))
            face = image[y1:y2, x1:x2, :]
            outImageName = os.path.join(path, fileName,
                                        fileName + '_' + str(i) + '.png')
            print(outImageName)
            imageio.imsave(outImageName, face)

            # #if not show:
            # plt.subplot(2, 6, 1)
            # plt.imshow(face)

            # face1 = misc.imresize(
            #     face, ((y2 - y1) * 2, (x2 - x1) * 2), interp='bilinear')
            # plt.subplot(2, 6, 2)
            # plt.imshow(face1)
            # plt.title('bilinear')

            # face1 = sharpen(face1, f)
            # plt.subplot(2, 6, 8)
            # plt.imshow(face1)

            # face1 = misc.imresize(
            #     face, ((y2 - y1) * 2, (x2 - x1) * 2), interp='nearest')
            # plt.subplot(2, 6, 3)
            # plt.imshow(face1)
            # plt.title('nearest')

            # face1 = sharpen(face1, f)
            # plt.subplot(2, 6, 9)
            # plt.imshow(face1)

            # face1 = misc.imresize(
            #     face, ((y2 - y1) * 2, (x2 - x1) * 2), interp='lanczos')
            # plt.subplot(2, 6, 4)
            # plt.imshow(face1)
            # plt.title('lanczos')

            # face1 = sharpen(face1, f)
            # plt.subplot(2, 6, 10)
            # plt.imshow(face1)

            # face1 = misc.imresize(
            #     face, ((y2 - y1) * 2, (x2 - x1) * 2), interp='bicubic')
            # plt.subplot(2, 6, 5)
            # plt.imshow(face1)
            # plt.title('bicubic')

            # face1 = sharpen(face1, f)
            # plt.subplot(2, 6, 11)
            # plt.imshow(face1)

            # face1 = misc.imresize(
            #     face, ((y2 - y1) * 2, (x2 - x1) * 2), interp='cubic')
            # plt.subplot(2, 6, 6)
            # plt.imshow(face1)
            # plt.title('cubic')

            # face1 = sharpen(face1, f)
            # plt.subplot(2, 6, 12)
            # plt.imshow(face1)

            # show = True
            # plt.show()

            i += 1
