import os
import sys
import shutil
import imageio
import numpy as np
from scipy import misc


def recursiveTraverse(searchPath, func, basePath, outPath):
    dirs = [
        os.path.join(searchPath, dir) for dir in os.listdir(searchPath)
        if os.path.isdir(os.path.join(searchPath, dir))
    ]
    for dir in dirs:
        recursiveTraverse(dir, func, basePath, outPath)
    files = [
        os.path.join(searchPath, dir) for dir in os.listdir(searchPath)
        if os.path.isfile(os.path.join(searchPath, dir))
    ]
    for file in files:
        func(file, os.path.join(outPath, os.path.relpath(file, basePath)))


def extendTo180(file, destFile):
    print(file, '-', destFile)
    if not os.path.exists(os.path.dirname(destFile)):
        os.makedirs(os.path.dirname(destFile))
    try:
        img = imageio.imread(file)
        imageSize = img.shape[0:2]
        if imageSize[0] != 180 or imageSize[1] != 180:
            n = np.random.randint(0, 255, (180, 180, 3), dtype=np.uint8)
            img = misc.imresize(img, (160, 160), 'bilinear')
            n[10:170, 10:170, :] = img
            img = n
            imageio.imsave(destFile, img)
        else:
            shutil.copy(file, destFile)
    except ValueError:
        pass


recursiveTraverse(sys.argv[1], extendTo180, sys.argv[1], sys.argv[2])
