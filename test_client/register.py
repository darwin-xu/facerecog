#!/usr/bin/env python
"""Upload files to server."""
import os
import sys
import requests
import cv2
import imgUtil


def upload_file():
    """Upload file."""
    username = ""
    folder = ""
    base_uri = 'http://127.0.0.1:5000'

    if len(sys.argv) != 3:
        print("Usage: Please provide <username> and <upload_folder>.")
        sys.exit(1)
    else:
        username = sys.argv[1]
        folder = sys.argv[2]

    # Iterate current folder and upload to server
    files = (os.listdir(folder))
    # Upload files to server
    counter = 0
    url_register = base_uri + "/registerFace/" + username
    for f in files:
        fileFullName = os.path.join(folder, f)

        thumbnail = imgUtil.resize_image(fileFullName)
        if thumbnail is None:
            continue
            
        result = cv2.imencode('.jpg', thumbnail)[1].tostring()
        files = {'file': result}
        response = requests.post(url_register, files=files)

        counter += 1

        print("Uploading... file [{}] {}".format(counter, fileFullName))
        if response.ok:
            print("SUCCESS\n")
        else:
            print(response)


upload_file()
