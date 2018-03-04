#!/usr/bin/env python
"""Client for detect face with name and indicated rectangle."""
import os, sys, platform
import json
from time import sleep
import requests
import base64

from PIL import Image, ImageFont, ImageDraw, ImageEnhance

def detect():
    # init_cmd_params()

    img   = ""

    if len(sys.argv) != 2:
        print ("Usage: Please provide <fileName>.")
        sys.exit(1)
    else:
        img = sys.argv[1]
        print(img)

    url_detect = 'http://127.0.0.1:5000/detectFace'

    with open(img, "rb") as imgFile:
        encoded_image = imgFile.read()

    files = {'file': encoded_image}

    response = requests.post(url_detect, files=files)

    print("get response")
    print(response)

    if response.ok:
        result = json.loads(response.content.decode('utf-8'))

        for f in result["result"]:
            print(f)
            if (f["possibility"] < 0.8):

                source_img = Image.open(img).convert("RGB")
                draw = ImageDraw.Draw(source_img)

                draw.rectangle(
                    ((f["x1"], f["y1"]), (f["x2"], f["y2"])),
                    fill = None,
                    outline = "red"
                )


                draw.text(
                    (f["x1"], f["y2"]),
                    f["name"]
                )

                # TODO: Get the absolute path
                #out_file=os.path.expanduser("~/Downloads/upload/result.jpg")
                out_file=("./result.jpg")
                source_img.save(out_file, "JPEG")

                Image.open(out_file).convert("RGBA").show()

                print("Matched!!!")
            else:
                print("Not match...")

            sleep(3)

        # faceInfo = [FaceInfo(**f) for f in result["result"]]
        # print(faceInfo)

detect()



