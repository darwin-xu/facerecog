#!/usr/bin/env python
"""Client for detect face with name and indicated rectangle."""
import sys
import json
import requests
import cv2
import imgUtil
import os


def detect():
    """Detect face function."""
    img = ""

    # Default interface
    base_URL = 'http://49.4.15.32:5000/'
    # base_URL = 'http://127.0.0.1:5000/'
    detect_URL = 'detectFacesC'

    if len(sys.argv) < 2:
        print("Usage: Please provide [detect method] <fileName>.")
        print("   -c for detectFacesC")
        print("   -d for detectFacesD")
        sys.exit(1)
    else:
        for i in range(1, len(sys.argv)):
            if sys.argv[i].startswith('-c'):
                detect_URL = 'detectFacesC'
            elif sys.argv[i].startswith('-d'):
                detect_URL = 'detectFacesD'
            else:
                img = sys.argv[i]
                filename_withoutext = os.path.splitext(sys.argv[i])[0]

    # Resize image in order to handle it smoothly
    thumbnail = imgUtil.resize_image(img)
    result = cv2.imencode('.jpg', thumbnail)[1].tostring()
    files = {'file': result}

    response = requests.post(base_URL + detect_URL, files=files)

    #print(response)

    if response.ok:
        result = json.loads(response.content.decode('utf-8'))
        print(result)

        for f in result[0]["faces"]:
            print(f)
            cv2.rectangle(thumbnail, (f["x1"], f["y1"]),
                            (f["x2"], f["y2"]), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            # Add name on top of the rectangle
            poss = f["possibility"]
            cv2.putText(thumbnail, f["id"] + " " + f'{poss:.2f}', (f["x1"], f["y2"] + 30), font,
                        1, (255, 255, 255), 1, cv2.LINE_AA)


        cv2.imshow("image", thumbnail)
        cv2.imwrite(filename_withoutext + "_dt.jpg", thumbnail)


detect()

if cv2.waitKey(0) & 0xFF == ord('q'):
    print("release resoure...")
    cv2.destroyAllWindows()
