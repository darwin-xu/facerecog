from __future__ import absolute_import, division, print_function

import argparse
import copy
import math
import os
import os.path
import pickle
import sys
import time
from functools import wraps
from os.path import join as pjoin
from sys import argv

import matplotlib.pyplot as plt
import numpy as np
import scipy
import tensorflow as tf
from scipy import misc
from sklearn.externals import joblib
from sklearn.svm import SVC

import detect_face
import facenet


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time.time()
        result = f(*args, **kwds)
        elapsed = time.time() - start
        print("%s took %.3f second to finish" % (f.__name__, elapsed))
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
    result = encode_faces(graph, sess, pnet, rnet, onet, image)
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
    return pos, bbs, rec_ids


def distance(emb1, emb2):
    dist = np.sum(np.square(np.subtract(emb1, emb2)))
    return dist


def distance2Possibility(distance, threshold):
    p = 2
    a = 0.4 / threshold**p
    return max(1 - a * distance**p, 0)


@timed
def search_face_by_distance(embeddings, tofind, threshold):
    min_dist = 1000.0
    min_id = ''

    for id, embs in embeddings.items():
        for emb in embs:
            dist = distance(emb, tofind)
            if (dist < min_dist):
                min_dist = dist
                min_id = id

    poss = 0
    if (min_dist < threshold):
        poss = distance2Possibility(min_dist, threshold)
    else:
        min_id = ""

    return min_id, poss


def thumbnailDetect(image, pnet, rnet, onet):
    minsize = 20  # minimum size of face
    threshold = [0.5, 0.6, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor
    shrink = 0.3

    thumbnail = misc.imresize(image, shrink, interp='bilinear')

    bounding_boxes, _ = detect_face.detect_face(thumbnail, minsize, pnet, rnet,
                                                onet, threshold, factor)
    return bounding_boxes / shrink


def computeEmbedding(graph, sess, images, batch_size=100):
    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")

    embedding_size = embeddings.get_shape()[1]
    nrof_images = len(images)
    emb_array = np.zeros((nrof_images, embedding_size))
    nrof_batches = int(math.ceil(1.0 * nrof_images / batch_size))

    for i in range(nrof_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, nrof_images)
        feed_dict = {
            images_placeholder: images[start_index:end_index],
            phase_train_placeholder: False
        }
        emb_array[start_index:end_index, :] = sess.run(
            embeddings, feed_dict=feed_dict)

    return emb_array


@timed
def encode_faces(graph, sess, pnet, rnet, onet, image):
    emb_face_size = 160
    tra_face_size = 180

    if image.ndim == 2:
        image = facenet.to_rgb(image)
    image = image[:, :, 0:3]
    image_size = np.asarray(image.shape)[0:2]
    bounding_boxes = thumbnailDetect(image, pnet, rnet, onet)

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

    if len(emb_img_list) > 0:
        images = np.stack(emb_img_list)

        embs = computeEmbedding(graph, sess, images)

        for i in range(len(embs)):
            result.append((embs[i], emb_boxes[i], tra_img_list[i]))

    return result


def freshEmbedding(graph, sess, pathOfImage):
    imageIds = [
        folder for folder in os.listdir(pathOfImage)
        if os.path.isdir(os.path.join(pathOfImage, folder))
    ]

    embeddings = {}

    for id in imageIds:
        imageFolder = os.path.join(pathOfImage, id)
        imagePaths = [
            os.path.join(imageFolder, file) for file in os.listdir(imageFolder)
            if os.path.isfile(os.path.join(imageFolder, file))
        ]
        images = facenet.load_data(imagePaths, False, False, 160)
        embs = computeEmbedding(graph, sess, images)
        embeddings[id] = []
        for emb in embs:
            embeddings[id].append(emb)

    return embeddings


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
