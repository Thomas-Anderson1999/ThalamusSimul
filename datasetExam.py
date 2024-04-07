import cv2
from ThalamusEngine.Interface import *
from ThalamusDSloader import *

if __name__ == '__main__':

    datasetPath = "dataset/"

    image_FileName, groundtruth_Filename, depth_Filename, IMULog, MotionContolLog, RoverCmdLog, depth_Width, depth_Height = getMetadata(datasetPath=datasetPath, filename="meta.json")
    if image_FileName != None:
        print(image_FileName, groundtruth_Filename, depth_Filename, depth_Width, depth_Height)

        #GroundTruth = getGroundTruth(groundtruth_Filename)
        #print(GroundTruth)

        if True == LoadThalamusInterface():
            print("open Engine")

        cap = cv2.VideoCapture(image_FileName)
        imgcnt = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("No Cam")
                break

            #print("imgcnt:", imgcnt, "(GT)x,y,z,r,p,y:", GroundTruth[imgcnt][0], GroundTruth[imgcnt][1], GroundTruth[imgcnt][2], GroundTruth[imgcnt][3], GroundTruth[imgcnt][4], GroundTruth[imgcnt][5])
            cv2.imshow('image', image)

            Depth_Map = np.zeros((depth_Height, depth_Width), np.float32)
            Depth_Mask = np.zeros((depth_Height, depth_Width, 3), np.uint8)
            if 0 != LoadBinDepthMapPnt(depth_Filename, depth_Width, depth_Height, 0, 10000, Depth_Map.ctypes, Depth_Mask.ctypes, imgcnt):
                norm_Depth_Map = cv2.normalize(Depth_Map, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                cv2.imshow("norm_Depth_Map", norm_Depth_Map)


            imgcnt += 1
            if cv2.waitKey(5) & 0xFF == 27:
                break

        cap.release()

