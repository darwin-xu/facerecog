#!/usr/bin/env python
"""Upload files to server."""
import os
import sys
import requests
import cv2
import imgUtil

class ImageClass():
    "Stores the paths to images for a given class"
    def __init__(self, name, image_paths):
        self.name = name
        self.image_paths = image_paths
  
    def __str__(self):
        return self.name + ', ' + str(len(self.image_paths)) + ' images'
  
    def __len__(self):
        return len(self.image_paths)
  
def get_dataset(paths, has_class_directories=True):
    dataset = []
    for path in paths.split(':'):
        path_exp = os.path.expanduser(path)
        classes = os.listdir(path_exp)
        classes.sort()
        nrof_classes = len(classes)
        for i in range(nrof_classes):
            class_name = classes[i]
            facedir = os.path.join(path_exp, class_name)
            image_paths = get_image_paths(facedir)
            if (len(image_paths) > 10):
                dataset.append(ImageClass(class_name, image_paths))
  
    return dataset

def get_image_paths(facedir):
    image_paths = []
    if os.path.isdir(facedir):
        images = os.listdir(facedir)
        image_paths = [os.path.join(facedir,img) for img in images]
    return image_paths
  
def upload_files():
    if len(sys.argv) != 2:
        print("Usage: Please provide <upload_folder>.")
        sys.exit(1)
    else:
        folder = sys.argv[1]

    datadir = folder
    dataset = get_dataset(datadir)
    print('Number of classes: %d' % len(dataset))

    for data in dataset:
        upload_file(data.name, data.image_paths)
        
    upload_file_done()

def upload_file(username, image_paths):
    base_uri = 'http://127.0.0.1:5000'

    # Iterate current folder and upload to server
    # files = (os.listdir(folder))
    # Upload files to server
    counter = 0
    url_register = base_uri + "/registerFaces/" + username
    for f in image_paths:
        # fileFullName = os.path.join(folder, f)

        thumbnail = imgUtil.resize_image(f)
        result = cv2.imencode('.jpg', thumbnail)[1].tostring()
        files = {'file': result}
        response = requests.post(url_register, files=files)

        counter += 1

        print("Uploading... file [{}] {}".format(counter, f))
        if response.ok:
            print("SUCCESS\n")
        else:
            print(response)

def upload_file_done():
    base_uri = 'http://127.0.0.1:5000'

    url_register_done = base_uri + "/registerFacesDone"
   
    response = requests.post(url_register_done)

    if response.ok:
        print("SUCCESS\n")
    else:
        print(response)


upload_files()
