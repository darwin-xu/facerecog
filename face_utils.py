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
                (model, class_names) = pickle.load(infile)
                print('load classifier file-> %s' % classifier_filename_exp)
                print('class names:-> %s' % class_names)

    return model, sess, graph, class_names, pnet, rnet, onet

def recong_face(model, sess, graph, class_names, pnet, rnet, onet, imgpath):
    print(imgpath)

    minsize = 20  # minimum size of face
    threshold = [0.1, 0.6, 0.9]  # three steps's threshold
    factor = 0.709  # scale factor
    frame_interval = 3
    image_size = 182
    input_image_size = 160

    class_names = ['fbb','gai','pg1','zw','zzy']    #train human name

    images_placeholder = graph.get_tensor_by_name("input:0")
    embeddings = graph.get_tensor_by_name("embeddings:0")
    phase_train_placeholder = graph.get_tensor_by_name("phase_train:0")
    embedding_size = embeddings.get_shape()[1]

    c = 0

    print('Start Recognition!')
    prevTime = 0
    if True:
        frame = cv2.imread(imgpath)
        curTime = time.time()    # calc fps
        timeF = frame_interval

        if (c % timeF == 0):
            find_results = []

            if frame.ndim == 2:
                frame = facenet.to_rgb(frame)
            frame = frame[:, :, 0:3]
            bounding_boxes, _ = detect_face.detect_face(frame, minsize, pnet, rnet, onet, threshold, factor)
            nrof_faces = bounding_boxes.shape[0]
            print('Detected_FaceNum: %d' % nrof_faces)

            if nrof_faces > 0:
                det = bounding_boxes[:, 0:4]
                img_size = np.asarray(frame.shape)[0:2]

                cropped = []
                scaled = []
                scaled_reshape = []
                bb = np.zeros((nrof_faces,4), dtype=np.int32)

                for i in range(nrof_faces):
                    emb_array = np.zeros((1, embedding_size))

                    bb[i][0] = det[i][0]
                    bb[i][1] = det[i][1]
                    bb[i][2] = det[i][2]
                    bb[i][3] = det[i][3]

                    # inner exception
                    if bb[i][0] <= 0 or bb[i][1] <= 0 or bb[i][2] >= len(frame[0]) or bb[i][3] >= len(frame):
                        print('face is inner of range!')
                        continue

                    cropped.append(frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :])
                    cropped[i] = facenet.flip(cropped[i], False)
                    scaled.append(misc.imresize(cropped[i], (image_size, image_size), interp='bilinear'))
                    scaled[i] = cv2.resize(scaled[i], (input_image_size,input_image_size),
                                            interpolation=cv2.INTER_CUBIC)
                    scaled[i] = facenet.prewhiten(scaled[i])
                    scaled_reshape.append(scaled[i].reshape(-1,input_image_size,input_image_size,3))
                    feed_dict = {images_placeholder: scaled_reshape[i], phase_train_placeholder: False}
                    emb_array[0, :] = sess.run(embeddings, feed_dict=feed_dict)
                    predictions = model.predict_proba(emb_array)
                    best_class_indices = np.argmax(predictions, axis=1)
                    best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                    cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)    #boxing face

                    #plot result idx under box
                    text_x = bb[i][0]
                    text_y = bb[i][3] + 20
                    # print('result: ', best_class_indices[0])
                    for H_i in class_names:
                        if class_names[best_class_indices[0]] == H_i:
                            result_names = class_names[best_class_indices[0]]
                            print (result_names)
                            print (best_class_probabilities)
                            cv2.putText(frame, result_names + str(best_class_probabilities), (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                        1, (0, 0, 255), thickness=1, lineType=2)
            else:
                print('Unable to align')

        sec = curTime - prevTime
        prevTime = curTime
        # fps = 1 / (sec)
        # str_fps = 'FPS: %2.3f' % fps
        # text_fps_x = len(frame[0]) - 150
        # text_fps_y = 20
        # cv2.putText(frame, str_fps, (text_fps_x, text_fps_y),
        #             cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), thickness=1, lineType=2)
        c+=1
        newimgpath = os.path.splitext(imgpath)[0] + '2' + os.path.splitext(imgpath)[1];
        cv2.imwrite(newimgpath, frame)
        
    cv2.destroyAllWindows()
    return newimgpath

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
