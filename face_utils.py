from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from sys import argv
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
import scipy

def load_model(modeldir, classifier_filename):
    print('Creating networks and loading parameters')
    graph = tf.get_default_graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(sess, '../facenet/src/align')

            print('Loading feature extraction model')
            facenet.load_model(modeldir)

            classifier_filename_exp = os.path.expanduser(classifier_filename)
            with open(classifier_filename_exp, 'rb') as infile:
                (model, ids) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)
                print('class names:-> %s' % ids)

    return model, sess, graph, ids, pnet, rnet, onet

def recong_face_c(model, sess, graph, ids, pnet, rnet, onet, image):
    result = encode_faces(graph, sess, pnet, rnet, onet, image)
    pos = []
    bbs = []
    ids = []
    for r in result:
        emb_array, bb = r
        predictions = model.predict_proba(emb_array)
        best_class_indices = np.argmax(predictions, axis=1)
        best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
        pos += best_class_probabilities
        bbs += bb

        for H_i in ids:
            if ids[best_class_indices[0]] == H_i:
                id = ids[best_class_indices[0]]
                print (id)
                print (best_class_probabilities)
                ids += id
    return pos, bb, ids

def distance(emb1, emb2):
    dist = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
    return dist

def search_face_by_distance(embeddings, tofind):
    min_dist = 1000.0
    min_id = ''
    threshold = 1.2
    for id, embs in embeddings.items():
        for emb in embs:
            dist = distance(emb, tofind)
            if (dist < min_dist):
                min_dist = dist
                min_id = id

    if (min_dist < threshold):
        return min_id, min_dist

def encode_faces(graph, sess, pnet, rnet, onet, image):
    minsize = 20  # minimum size of face
    threshold = [0.1, 0.6, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor
    face_size = 160
    margin = 32

    if image.ndim == 2:
        image = facenet.to_rgb(image)
    image = image[:, :, 0:3]
    image_size = np.asarray(image.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet, onet, threshold, factor)
    result = []
    nrof_faces = bounding_boxes.shape[0]
    boxes = np.zeros((nrof_faces, 4), dtype=np.int32)
    img_list = []
    for i in range(nrof_faces):
        boxes[i][0] = np.maximum(bounding_boxes[i, 0] - margin / 2, 0)
        boxes[i][1] = np.maximum(bounding_boxes[i, 1] - margin / 2, 0)
        boxes[i][2] = np.minimum(bounding_boxes[i, 2] + margin / 2, image_size[1])
        boxes[i][3] = np.minimum(bounding_boxes[i, 3] + margin / 2, image_size[0])

        #scipy.misc.imsave('out.jpg', image)
        cropped = image[boxes[i][1]:boxes[i][3], boxes[i][0]:boxes[i][2], :]
        aligned = misc.imresize(cropped, (face_size, face_size), interp='bilinear')
        #scipy.misc.imsave('out_' + str(i) + '.jpg', aligned)
        prewhitened = facenet.prewhiten(aligned)
        img_list.append(prewhitened)

    images = np.stack(img_list)

    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

    feed_dict = { images_placeholder: images, phase_train_placeholder: False }
    embs = sess.run(embeddings, feed_dict=feed_dict)

    for i in range(len(embs)):
       result.append((embs[i], boxes[i]))

    return result

def main(argv=None):
    if argv is None:
        argv = sys.argv
    model, sess, graph, class_names, pnet, rnet, onet = load_model('C:\\Users\\edaiwxu\\Documents\\projects\\models\\20170512-110547',
        'C:\\Users\\edaiwxu\\Documents\\projects\\models\\my_classifier.pkl')
    #test_image(model, sess, graph, class_names, pnet, rnet, onet, argv[1])
    frame = cv2.imread("C:\\Users\\edaiwxu\\Downloads\\IMG_9473.jpg")
    result = encode_faces(graph, sess, pnet, rnet, onet, frame)
    print("faces detect:")
    print(len(result))
    for r in result:
        e, box = r
        print(e)

if __name__ == "__main__":
    sys.exit(main())