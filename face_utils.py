from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
from sys import argv
import tensorflow as tf
from scipy import misc
import matplotlib.pyplot as plt
import numpy as np
import argparse
import facenet
import detect_face
import os
from os.path import join as pjoin
import sys
import time
from functools import wraps
import copy
import math
import pickle
from sklearn.svm import SVC
from sklearn.externals import joblib
import scipy


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time.time()
        result = f(*args, **kwds)
        elapsed = time.time() - start
        print("%s took %f second to finish" % (f.__name__, elapsed))
        return result

    return wrapper


def randomSelect(items, count):
    idx = np.random.choice(
        len(items), np.minimum(len(items), count), replace=False)
    ret = []
    for i in idx:
        ret.append(items[i])
    return ret


def crossCheckArray(emb1, emb2, fullCheck, subsetSize):
    embs1 = []
    embs2 = []
    actual_issame = []
    if emb1 is emb2:
        if not fullCheck:
            emb1 = randomSelect(emb1, subsetSize)
        for i in range(len(emb1) - 1):
            for j in range(i + 1, len(emb1)):
                embs1.append(emb1[i])
                embs2.append(emb1[j])
                actual_issame.append(True)
    else:
        if not fullCheck:
            emb1 = randomSelect(emb1, subsetSize)
            emb2 = randomSelect(emb2, subsetSize)
        for i in range(len(emb1)):
            for j in range(len(emb2)):
                embs1.append(emb1[i])
                embs2.append(emb2[j])
                actual_issame.append(False)
    return embs1, embs2, actual_issame


def crossCheckDict(embeddings, fullCheck=False):
    total_embs1 = []
    total_embs2 = []
    total_actual_issame = []
    keys = list(embeddings.keys())
    if not fullCheck:
        keys = randomSelect(keys, 100)
    for i in range(len(keys)):
        for j in range(i, len(keys)):
            embs1, embs2, actual_issame = crossCheckArray(
                embeddings[keys[i]], embeddings[keys[j]], fullCheck,
                int(400 / len(keys)))
            total_embs1 += embs1
            total_embs2 += embs2
            total_actual_issame += actual_issame

    return np.stack(total_embs1), np.stack(total_embs2), total_actual_issame


def load_model(modeldir, classifier_filename):
    print('Creating networks and loading parameters')
    graph = tf.get_default_graph()
    with graph.as_default():
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.3)
        sess = tf.Session(
            config=tf.ConfigProto(
                gpu_options=gpu_options, log_device_placement=False))
        with sess.as_default():
            pnet, rnet, onet = detect_face.create_mtcnn(
                sess, '../facenet/src/align')

            print('Loading feature extraction model')
            facenet.load_model(modeldir)

            model = None
            classifier_filename_exp = os.path.expanduser(classifier_filename)
            if os.path.exists(classifier_filename_exp):
                with open(classifier_filename_exp, 'rb') as infile:
                    model = pickle.load(infile)
                    print(
                        'load classifier file-> %s' % classifier_filename_exp)

    return model, sess, graph, pnet, rnet, onet


@timed
def recong_face_c(model, sess, graph, pnet, rnet, onet, image):
    now = int(round(time.time() * 1000))
    result = encode_faces(graph, sess, pnet, rnet, onet, image)
    last = int(round(time.time() * 1000))
    print("detect face cost: " + str(last - now) + " milliseconds")
    # print ("the result array's shape: " + str(emb_array.shape))
    pos = []
    bbs = []
    rec_ids = []
    for r in result:
        emb, bb, _ = r
        embedding_size = emb.shape[0]
        emb_array = np.zeros((1, embedding_size))
        emb_array[0, :] = emb
        predictions = model.predict_proba(emb_array)
        # print (predictions)
        posibs = np.argpartition(predictions[0], -2)[-2:]
        print(model.classes_[posibs[1]])
        print(predictions[0][posibs[1]])
        print(model.classes_[posibs[0]])
        print(predictions[0][posibs[0]])
        if (predictions[0][posibs[1]] > predictions[0][posibs[0]] * 2.0):
            print('yes')
            # best_class_indices = np.argmax(predictions, axis=1)
            # print (best_class_indices)
            best_class_probabilities = predictions[0][posibs[1]]
            pos.append(best_class_probabilities)
            bbs.append(bb)
            rec_ids.append(model.classes_[posibs[1]])
    print("recong face c cost: " + str(int(round(time.time() * 1000)) - last) +
          " milliseconds")
    return pos, bbs, rec_ids


def distance(emb1, emb2):
    dist = np.sqrt(np.sum(np.square(np.subtract(emb1, emb2))))
    return dist


@timed
def search_face_by_distance(embeddings, tofind, threshold):
    min_dist = 1000.0
    min_id = ''
    #threshold = 0.51
    for id, embs in embeddings.items():
        for emb in embs:
            dist = distance(emb, tofind)
            if (dist < min_dist):
                min_dist = dist
                min_id = id

    posi = 0
    if (min_dist < threshold):
        posi = 1 - min_dist / threshold
    else:
        min_id = "unknown"

    return min_id, min_dist


@timed
def encode_faces(graph, sess, pnet, rnet, onet, image):
    minsize = 20  # minimum size of face
    threshold = [0.5, 0.6, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor
    emb_face_size = 160
    tra_face_size = 180

    if image.ndim == 2:
        image = facenet.to_rgb(image)
    image = image[:, :, 0:3]

    image_size = np.asarray(image.shape)[0:2]
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet,
                                                onet, threshold, factor)

    result = []
    nrof_faces = bounding_boxes.shape[0]
    emb_boxes = np.zeros((nrof_faces, 4), dtype=np.int32)
    emb_img_list = []
    tra_img_list = []
    for i in range(nrof_faces):
        emb_boxes[i][0] = np.maximum(bounding_boxes[i, 0], 0)  # x1
        emb_boxes[i][1] = np.maximum(bounding_boxes[i, 1], 0)  # y1
        emb_boxes[i][2] = np.minimum(bounding_boxes[i, 2], image_size[1])  # x2
        emb_boxes[i][3] = np.minimum(bounding_boxes[i, 3], image_size[0])  # y2

        margin = (bounding_boxes[i, 2] - bounding_boxes[i, 0]) / 8
        tra_box = np.zeros((4, ), dtype=np.int32)

        tra_box[0] = np.maximum(bounding_boxes[i, 0] - margin / 2, 0)  # x1
        tra_box[1] = np.maximum(bounding_boxes[i, 1] - margin / 2, 0)  # y1
        tra_box[2] = np.minimum(bounding_boxes[i, 2] + margin / 2,
                                image_size[1])  # x2
        tra_box[3] = np.minimum(bounding_boxes[i, 3] + margin / 2,
                                image_size[0])  # y2

        emb_cropped = image[emb_boxes[i][1]:emb_boxes[i][3], emb_boxes[i][0]:
                            emb_boxes[i][2], :]
        tra_cropped = image[tra_box[1]:tra_box[3], tra_box[0]:tra_box[2], :]
        emb_aligned = misc.imresize(
            emb_cropped, (emb_face_size, emb_face_size), interp='bilinear')
        tra_aligned = misc.imresize(
            tra_cropped, (tra_face_size, tra_face_size), interp='bilinear')
        prewhitened = facenet.prewhiten(emb_aligned)
        emb_img_list.append(prewhitened)
        tra_img_list.append(tra_aligned)

    images = np.stack(emb_img_list)

    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

    feed_dict = {images_placeholder: images, phase_train_placeholder: False}
    embs = sess.run(embeddings, feed_dict=feed_dict)

    for i in range(len(embs)):
        result.append((embs[i], emb_boxes[i], tra_img_list[i]))

    return result


def generate_response(posbs, bbs, recg_ids):
    response = {}
    response['faces'] = []
    for i in range(len(posbs)):
        face = {}
        face['possibility'] = posbs[i]
        face['x1'] = bbs[i][0]
        face['y1'] = bbs[i][1]
        face['x2'] = bbs[i][2]
        face['y2'] = bbs[i][3]
        face['id'] = recg_ids[i]
        response['faces'].append(face)

    print(response)
    return response


def main(argv=None):
    if argv is None:
        argv = sys.argv
    model, sess, graph, ids, pnet, rnet, onet = load_model(
        '../models/20170511-185253', '../models/my_classifier.pkl')
    #test_image(model, sess, graph, class_names, pnet, rnet, onet, argv[1])
    frame = imageio.imread('./3p.jpg')
    print(frame.shape)
    pos, bbs, rec_ids = recong_face_c(model, sess, graph, ids, pnet, rnet,
                                      onet, frame)
    print(pos)
    print(bbs)
    print(rec_ids)
    print(generate_response(pos, bbs, rec_ids))


if __name__ == "__main__":
    sys.exit(main())
