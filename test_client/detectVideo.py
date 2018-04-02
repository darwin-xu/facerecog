import numpy as np
import cv2
import requests
import time
from param import base_uri
import json

detect_uri = '/detectFacesD'

cap = cv2.VideoCapture(0)

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        m1 = int(round(time.time() * 1000))
        result = cv2.imencode('.jpg', frame)[1].tostring()
        files = {'file': result}
        m2 = int(round(time.time() * 1000))
        response = requests.post(base_uri + detect_uri, files=files)
        m3 = int(round(time.time() * 1000))

        if response.ok:
            result = json.loads(response.content.decode('utf-8'))

            for f in result[0]["faces"]:
                print(f)
                cv2.rectangle(frame, (f["x1"], f["y1"]), (f["x2"], f["y2"]),
                              (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                # Add name on top of the rectangle
                poss = f["possibility"]
                cv2.putText(frame, f["id"] + " " + f'{poss:.2f}',
                            (f["x1"], f["y2"] + 30), font, 1, (255, 255, 255),
                            1, cv2.LINE_AA)
        m4 = int(round(time.time() * 1000))
        print("step 1", m2 - m1)
        print("step 2", m3 - m2)
        print("step 3", m4 - m3)
    elif key & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('frame', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
