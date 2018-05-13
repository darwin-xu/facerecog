import tensorflow as tf
import numpy as np
import cv2
import detect_face
cap = cv2.VideoCapture(0)

freeze = False


def thumbnailDetect(image, pnet, rnet, onet):
    image_size = np.asarray(image.shape)[0:2]
    minsize = image_size[0] * 0.07  # minimum size of face
    minsize = 20 if minsize < 20 else minsize
    threshold = [0.6, 0.7, 0.7]  # three steps's threshold
    factor = 0.709  # scale factor
    bounding_boxes, _ = detect_face.detect_face(image, minsize, pnet, rnet,
                                                onet, threshold, factor)
    return bounding_boxes


def drawBox(image, box):
    width = box[2] - box[0]
    margin = width / 8
    innerBox = np.zeros((4, ), dtype=np.int32)
    innerBox[0] = box[0]
    innerBox[1] = box[1]
    innerBox[2] = box[2]
    innerBox[3] = box[3]
    outerBox = np.zeros((4, ), dtype=np.int32)
    outerBox[0] = box[0] - margin / 2
    outerBox[1] = box[1] - margin / 2
    outerBox[2] = box[2] + margin / 2
    outerBox[3] = box[3] + margin / 2
    cv2.rectangle(frame, (innerBox[0], innerBox[1]),
                  (innerBox[2], innerBox[3]), (0, 0, 255), 1)
    cv2.rectangle(frame, (outerBox[0], outerBox[1]),
                  (outerBox[2], outerBox[3]), (0, 0, 255), 1)


graph = tf.get_default_graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,
                                                    '../facenet/src/align')

        while (True):
            # Capture frame-by-frame
            _, frame = cap.read()

            bbs = thumbnailDetect(frame, pnet, rnet, onet)

            for bb in bbs:
                drawBox(frame, bb)
                if not freeze:
                    print(
                        '%1.3f' % ((bb[2] - bb[0]) / (bb[3] - bb[1])),
                        end='\r',
                        flush=True)

            if not freeze:
                cv2.imshow('frame', frame)

            key = cv2.waitKey(1) & 0xFF

            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('f') or key == ord('F'):
                freeze = not freeze
