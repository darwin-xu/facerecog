import sys
import time

import tensorflow as tf
from scipy import misc

import cv2
import detect_face

graph = tf.get_default_graph()
with graph.as_default():
    sess = tf.Session()
    with sess.as_default():
        pnet, rnet, onet = detect_face.create_mtcnn(sess,
                                                    '../facenet/src/align')

        pt = 0.5
        rt = 0.6
        ot = 0.9
        shrink = 1
        step = 0.3

        while True:
            image = cv2.imread(sys.argv[1])
            forshow = misc.imresize(image, 0.5, interp='bilinear')

            minsize = 20
            threshold = [pt, rt, ot]
            print(threshold)
            factor = 0.709

            # thumbnail = misc.imresize(
            #     image[:, :, ::-1], shrink, interp='bilinear')
            b = time.time()
            bounding_boxes, _ = detect_face.detect_face(
                image[:, :, ::-1], minsize, pnet, rnet, onet, threshold, factor)
            e = time.time()
            bounding_boxes = bounding_boxes / shrink * 0.5
            e = time.time()
            print('%.3f second' % (e - b))

            size = image.shape[0:2]
            print(bounding_boxes.shape)
            for x1, y1, x2, y2, _ in bounding_boxes:
                x1 = int(max(x1, 0))
                y1 = int(max(y1, 0))
                x2 = int(min(x2, size[1]))
                y2 = int(min(y2, size[0]))
                cv2.rectangle(forshow, (x1, y1), (x2, y2), (0, 0, 255), 1)

            cv2.imshow("image", forshow)

            key = cv2.waitKey(0) & 0xFF

            if key == ord('a'):
                pt -= step
            elif key == ord('z'):
                pt += step
            elif key == ord('s'):
                rt -= step
            elif key == ord('x'):
                rt += step
            elif key == ord('d'):
                ot -= step
            elif key == ord('c'):
                ot += step
            elif key == ord('q'):
                quit()
            elif key == ord('q'):
                pass

            cv2.destroyAllWindows()
