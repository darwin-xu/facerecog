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

def make_classifier(sess, graph, emb_array, classifier_filename):
    classifier_filename_exp = os.path.expanduser(classifier_filename)

    # Train classifier
    print('Training classifier')
    model = SVC(kernel='linear', probability=True)
    model.fit(emb_array.values(), emb_array.keys())

    # Saving classifier model
    with open(classifier_filename_exp, 'wb') as outfile:
        pickle.dump((model, emb_array.keys()), outfile)
    print('Saved classifier model to file "%s"' % classifier_filename_exp)
    print('Goodluck')