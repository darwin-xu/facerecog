import os
import sys

intDir = sys.argv[1]
outDir = sys.argv[2]

if not os.path.isdir(outDir):
    os.makedirs(outDir)

folders = [os.path.join(intDir, folder) for folder in os.listdir(intDir) if os.path.isdir(os.path.join(intDir, folder))]
for folder in folders:
    files = [file for file in os.listdir(folder)]
    i = 0
    #print(len(files))
    for file in files:
        srcFile = os.path.join(folder, file)
        srcFileDir = os.path.basename(os.path.normpath(folder))
        outFolder = os.path.join(outDir, srcFileDir)
        if not os.path.isdir(outFolder):
            os.makedirs(outFolder)
        dstFile = os.path.join(outDir, srcFileDir, file)
        print(srcFile, " ==> ", dstFile)
        i += 1
        if i > 4:
            break
        os.rename(srcFile, dstFile)
    print()
