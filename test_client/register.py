#!/usr/bin/env python
"""Upload files to server."""
import os
import sys
import requests


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

        with open(fileFullName, "rb") as image_file:
            encoded_image = image_file.read()

        files = {'file': encoded_image}

        response = requests.post(url_register, files=files)

        counter += 1

        print("Uploading... file [{}] {}".format(counter, fileFullName))
        if response.ok:
            print("SUCCESS\n")
        else:
            print(response)


upload_file()
