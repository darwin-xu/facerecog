import numpy as np
import cv2
import requests
import time
from param import base_uri
import json

detect_uri = '/detectFacesD'

cap = cv2.VideoCapture(0)

delay = 2000

result = None
last = int(round(time.time() * 1000))
while (True):
    now = int(round(time.time() * 1000))

    # Capture frame-by-frame
    ret, frame = cap.read()

    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)  #resize frame (optional)

    key = cv2.waitKey(1)
    if key & 0xFF == ord('c'):
        last = now
        m1 = int(round(time.time() * 1000))
        jpg = cv2.imencode('.jpg', frame)[1].tostring()
        files = {'file': jpg}
        response = requests.post(base_uri + detect_uri, files=files)
        m2 = int(round(time.time() * 1000))
        print("get response cost: " + str(m2 - m1) + " milliseconds")

        if response.ok:
            result = json.loads(response.content.decode('utf-8'))

    elif key & 0xFF == ord('q'):
        break

    if now - last > delay:
        result = None

    if result != None:
        for f in result[0]["faces"]:
            cv2.rectangle(frame, (f["x1"], f["y1"]), (f["x2"], f["y2"]),
                          (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Add name on top of the rectangle
            cv2.putText(frame, f["id"], (f["x1"], f["y2"] + 30), font, 1,
                        (255, 255, 255), 1, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('frame', frame)
    # i += 1
    # cv2.imwrite('out/messigray' + str(i) + '.jpg', frame)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
