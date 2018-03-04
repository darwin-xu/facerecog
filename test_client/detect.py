#!/usr/bin/env python
"""Client for detect face with name and indicated rectangle."""
import os
import sys
import platform
import json
# from time import sleep
import requests
from PIL import Image, ImageDraw, ImageFont

# from PIL import Image, ImageFont, ImageDraw, ImageEnhance


def detect():
    """Detect face function."""
    img = ""

    if len(sys.argv) != 2:
        print("Usage: Please provide <fileName>.")
        sys.exit(1)
    else:
        img = sys.argv[1]
        print(img)

    url_detect = 'http://127.0.0.1:5000/detectFacesC'

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
                    fill=None,
                    outline="red")

                # Define font based on different platform
                osType = platform.system()
                fontLoc = ""
                if (osType == "Window"):
                    # TODO: Update font location
                    fontLoc = "/Users/kevinzhong/Library/Fonts/SourceCodePro-Regular.ttf"
                elif (osType == "Darwin"):
                    fontLoc = "/Users/kevinzhong/Library/Fonts/SourceCodePro-Regular.ttf"
                elif (osType == "Linux"):
                    # TODO: Update font location
                    fontLoc = "/Users/kevinzhong/Library/Fonts/SourceCodePro-Regular.ttf"

                fontType = ImageFont.truetype(fontLoc, 18)

                draw.text(
                    (f["x1"], f["y2"]), f["name"], (0, 0, 255), font=fontType)

                # TODO: Get the absolute path
                out_file = os.path.expanduser("~/result.jpg")
                source_img.convert("RGB").save(out_file, "JPEG")

                Image.open(out_file).convert("RGB").show()

                print("Matched!!!")
            else:
                print("Not match...")

        # faceInfo = [FaceInfo(**f) for f in result["result"]]
        # print(faceInfo)


detect()
