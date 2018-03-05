from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import argparse
import facenet
import detect_face
import os
import sys
import math
import pickle
from sklearn.svm import SVC

def make_classifier(sess, graph, emb_dict, classifier_filename):
    # print('FKFKFKFKFKFKKFKFKFKFKFKFKFKFKKFKFKFKFKFKFKKFKFKFKKFKFKFKFKKFKFKFKKFKFKFK')
    classifier_filename_exp = os.path.expanduser(classifier_filename)

    nrof_embs = 0 
    labels = []
    emb_size = 0
    for emb in emb_dict.values():
        nrof_embs += len(emb)
        emb_size = len(emb[0])

    i = 0
    emb_array = np.zeros((nrof_embs, emb_size))
    for label, embs in emb_dict.items():
        for emb in embs:
            labels.append(label)
            emb_array[i, :] = emb
            i = i + 1

    # Train classifier
    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(emb_array, labels)

    classes = list(emb_dict.keys())
    # Saving classifier model
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, classes), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)
    print('Goodluck')

    return model, classes