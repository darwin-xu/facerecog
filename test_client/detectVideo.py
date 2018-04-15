import numpy as np
import cv2
import requests
import time
from param import base_uri
import json

detect_uri = '/detectFacesD'

cap = cv2.VideoCapture(0)

detecting = False


def putText(frame, text, point, scale):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, text, point, font, scale, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, text, point, font, scale, (255, 255, 255), 1,
                cv2.LINE_AA)


while (True):
    # Capture frame-by-frame
    _, frame = cap.read()

    key = cv2.waitKey(1) & 0xFF

    if key == ord('d') or key == ord('D'):
        detecting = True
    elif key == ord('s') or key == ord('S'):
        detecting = False
    elif key == ord('q') or key == ord('Q'):
        break

    putText(
        frame,
        "Press 'd' to start detect. Press 's' to stop detect. Press 'q' to quit",
        (0, 20), 0.5)

    if detecting:
        jpg = cv2.imencode('.jpg', frame)[1].tostring()
        files = {'file': jpg}
        response = requests.post(base_uri + detect_uri, files=files)
        if response.ok:
            result = json.loads(response.content.decode('utf-8'))
            for f in result[0]["faces"]:
                cv2.rectangle(frame, (f["x1"], f["y1"]), (f["x2"], f["y2"]),
                              (0, 0, 255), 2)
                # Add name on top of the rectangle
                poss = f["possibility"] * 100
                if poss >= 60:
                    tag = f["id"] + f' {poss:.1f}%'
                else:
                    tag = 'unknown'
                putText(frame, tag, (f["x1"], f["y2"] + 30), 0.7)

    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
