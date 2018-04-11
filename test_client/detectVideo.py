import numpy as np
import cv2
import requests
import time
from param import base_uri
import json

detect_uri = '/detectFacesD'

cap = cv2.VideoCapture(0)

detecting = False

while (True):
    # Capture frame-by-frame
    _, frame = cap.read()

    key = cv2.waitKey(1)

    if key & 0xFF == ord('d'):
        detecting = True
    elif key & 0xFF == ord('s'):
        detecting = False
    elif key & 0xFF == ord('q'):
        break

    if detecting:
        jpg = cv2.imencode('.jpg', frame)[1].tostring()
        files = {'file': jpg}
        response = requests.post(base_uri + detect_uri, files=files)
        if response.ok:
            result = json.loads(response.content.decode('utf-8'))
            for f in result[0]["faces"]:
                cv2.rectangle(frame, (f["x1"], f["y1"]), (f["x2"], f["y2"]),
                              (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Add name on top of the rectangle
                cv2.putText(frame, f["id"], (f["x1"], f["y2"] + 30), font, 1,
                            (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
