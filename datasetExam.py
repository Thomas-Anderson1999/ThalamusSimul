import cv2
import numpy as np
import json
from ThalamusEngine.Interface import *
def getGroundTruth(filename):
    res = []
    with open(filename, "r") as f:
        lines = f.readlines()
        for line in lines:
            #print(line)
            tmpLins = line.split("\t")
            res.append([float(tmp) for tmp in tmpLins])
    return np.array(res)

if __name__ == '__main__':

    datasetPath = "tmpDataset/"

    with open(datasetPath+"meta.json", "r") as fpjson:
        datasetMeta = json.load(fpjson)
        #print(datasetMeta)

        image_FileName = datasetPath + datasetMeta["image"]
        groundtruth_Filename = datasetPath + datasetMeta["groundtruth"]
        depth_Filename = datasetPath + datasetMeta["depthmapFile"]
        depth_Width = datasetMeta["DepthWidth"]
        depth_Height = datasetMeta["DepthHeight"]
        print(image_FileName, groundtruth_Filename, depth_Filename, depth_Width, depth_Height)

        GroundTruth = getGroundTruth(groundtruth_Filename)
        #print(GroundTruth)

        if True == LoadThalamusInterface():
            print("aaa")


        cap = cv2.VideoCapture(image_FileName)
        imgcnt = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("No Cam")
                break

            print("imgcnt:", imgcnt, "(GT)x,y,z,r,p,y:", GroundTruth[imgcnt][0], GroundTruth[imgcnt][1], GroundTruth[imgcnt][2], GroundTruth[imgcnt][3], GroundTruth[imgcnt][4], GroundTruth[imgcnt][5])

            imgcnt += 1
            cv2.imshow('image', image)

            Depth_Map = np.zeros((depth_Height, depth_Width), np.float32)
            Depth_Mask = np.zeros((depth_Height, depth_Width, 3), np.uint8)
            if 0 != LoadBinDepthMapPnt(depth_Filename, depth_Width, depth_Height, 0, 10000, Depth_Map.ctypes, Depth_Mask.ctypes, imgcnt):
                Depth_Map = cv2.normalize(Depth_Map, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                cv2.imshow("Depth Map", Depth_Map)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

