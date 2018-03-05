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
        print (result)

        source_img = Image.open(img).convert("RGB")
        for f in result[0]["faces"]:
            print(f)
            if (f["possibility"] < 0.8):

                draw = ImageDraw.Draw(source_img)

                draw.rectangle(
                    ((f["x1"], f["y1"]), (f["x2"], f["y2"])),
                    fill=None,
                    outline="red")

                # Define font based on different platform
                osType = platform.system()
                print(osType)
                fontLoc = ""
                if (osType == "Windows"):
                    fontLoc = "C:\\Windows\Fonts\\Arial\\ariblk.ttf"
                elif (osType == "Darwin"):
                    fontLoc = "/Users/kevinzhong/Library/Fonts/SourceCodePro-Regular.ttf"
                elif (osType == "Linux"):
                    fontLoc = "/usr/share/fonts/truetype/ubuntu-font-family/UbuntuMono-R.ttf"

                fontType = ImageFont.truetype(fontLoc, 18)

                draw.text(
                    (f["x1"], f["y2"]), f["id"], (0, 0, 255), font=fontType)

                # TODO: Get the absolute path
                #out_file=os.path.expanduser("~/Downloads/upload/result.jpg")


                print("Matched!!!")
            else:
                print("Not match...")

        out_file=("./result.jpg")
        source_img.save(out_file, "JPEG")
        Image.open(out_file).convert("RGB").show()

        # faceInfo = [FaceInfo(**f) for f in result["result"]]
        # print(faceInfo)


detect()
