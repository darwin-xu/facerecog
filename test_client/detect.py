#!/usr/bin/env python
"""Client for detect face with name and indicated rectangle."""
import sys
import json
import requests
import cv2
import imgUtil


def detect():
    """Detect face function."""
    img = ""

    # Default interface
    url_detect = 'http://127.0.0.1:5000/detectFacesC'

    if len(sys.argv) < 2:
        print("Usage: Please provide [detect method] <fileName>.")
        print("   -c for detectFacesC")
        print("   -d for detectFacesD")
        sys.exit(1)
    else:
        for i in range(1, len(sys.argv)):
            if sys.argv[i].startswith('-c'):
                url_detect = 'http://127.0.0.1:5000/detectFacesC'
            elif sys.argv[i].startswith('-d'):
                url_detect = 'http://127.0.0.1:5000/detectFacesD'
            else:
                img = sys.argv[i]

    # Resize image in order to handle it smoothly
    thumbnail = imgUtil.resize_image(img)
    result = cv2.imencode('.jpg', thumbnail)[1].tostring()
    files = {'file': result}

    response = requests.post(url_detect, files=files)

    print(response)

    if response.ok:
        result = json.loads(response.content.decode('utf-8'))
        print(result)

        for f in result[0]["faces"]:
            print(f)
            if (f["possibility"] < 0.8):

                cv2.rectangle(thumbnail, (f["x1"], f["y1"]),
                              (f["x2"], f["y2"]), (0, 0, 255), 2)
                # Add name on top of the rectangle
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(thumbnail, f["id"], (f["x1"], f["y2"] + 30), font,
                            1, (255, 0, 0), 1, cv2.LINE_AA)

                print("Matched!!!")
            else:
                print("Not match...")

        cv2.imshow("image", thumbnail)


detect()

if cv2.waitKey(0) & 0xFF == ord('q'):
    print("release resoure...")
    cv2.destroyAllWindows()
