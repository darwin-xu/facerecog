#!/usr/bin/env python
"""Upload files to server."""
import os
import sys
import requests
from PIL import Image


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

        max_height = 1024
        max_width = 1024
        with Image.open(fileFullName) as img:
            width, height = img.size
            print("original size:")
            print(img.size)

            # only shrink if img is bigger than required
            if max_height < height or max_width < width:
                # get scaling factor
                scaling_factor = max_height / float(height)
                if max_width / float(width) < scaling_factor:
                    scaling_factor = max_width / float(width)
                    # resize image
                    img = img.resize((int(width * scaling_factor),
                                      int(height * scaling_factor)),
                                     Image.ANTIALIAS)

            print("Scaled size:")
            print(img.size)
            output = img.tobytes()
            files = {'file': output}
            response = requests.post(url_register, files=files)

        counter += 1

        print("Uploading... file [{}] {}".format(counter, fileFullName))
        if response.ok:
            print("SUCCESS\n")
        else:
            print(response)


upload_file()
