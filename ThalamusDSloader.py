import numpy as np
import json

def getGroundTruth(filename):
    res = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            #print(line)
            tmpLins = line.split("\t")
            res.append([float(tmp) for tmp in tmpLins])
    return np.array(res)

def getMetadata(datasetPath, filename):
    try:
        with open(datasetPath+filename, "r") as fpjson:
            datasetMeta = json.load(fpjson)
            #print(datasetMeta)

            image_FileName = datasetPath + datasetMeta["image"]
            groundtruth_Filename = datasetPath + datasetMeta["groundtruth"]
            depth_Filename = datasetPath + datasetMeta["depthmapFile"]
            depth_Width = datasetMeta["DepthWidth"]
            depth_Height = datasetMeta["DepthHeight"]
            #print(image_FileName, groundtruth_Filename, depth_Filename, depth_Width, depth_Height)
            return image_FileName, groundtruth_Filename, depth_Filename, depth_Width, depth_Height
    except:
        return None, None, None, None, None
