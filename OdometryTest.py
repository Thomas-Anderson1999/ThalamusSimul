import cv2
from ThalamusEngine.Interface import *
from ThalamusDSloader import *
import matplotlib.pyplot as plt
import math

#+Image Processing
from ThalamusNavigation.ThalamusCV.ThalamusCV.cvContext import *
from ThalamusNavigation.ThalamusCV.ThalamusCV.cvUtil import *
from ThalamusNavigation.ThalamusCV.cvNavInterface import *
#-Image Processing

class cInterval:
    def __init__(self):
        self.startIdx = 0 #멈춰있는 기간을 포함한 시작점
        self.endIdx = 0 #멈춰있는 기간을 포함한 끝점
        self.ActionType = "none" #MV, RT, STOP
        self.ActionValue = 0 #실제로 간 거리
        self.AccTime = 0
        self.AccSpd = 0
        self.preIdx = 0
        self.ActualMvIdx = -1    # 실제로 움직이기 시작하는 idx
        self.ActualStopIdx = -1  # 실제로 멈춘 idx

    def show(self):
        print(self.startIdx, self.endIdx, self.preIdx, self.ActionType, self.ActionValue, self.AccTime, self.AccSpd)
def getPeriodfromRobotCmd(RoverCmdLog, preIndx, endIdx):
    res = []
    lens = len(RoverCmdLog[0])
    for i in range(lens):
        res.append(cInterval())
        if RoverCmdLog[0][i] - preIndx < 0:
            res[i].startIdx = 0
        else:
            res[i].startIdx = RoverCmdLog[0][i] - preIndx

        res[i].ActionType = RoverCmdLog[2][i]
        res[i].AccTime = RoverCmdLog[3][i]
        res[i].AccSpd = RoverCmdLog[4][i]
        res[i].ActionValue = RoverCmdLog[5][i]
        res[i].preIdx = preIndx

        res[i].endIdx = endIdx #update in next index
        if i != 0 :
            res[i-1].endIdx = res[i].startIdx
    return res



if __name__ == '__main__':
    if True == LoadThalamusInterface():
        print("open Engine")

    #+image process
    cutOffbboxOpt1 = [15, 15, 15, 15]
    featureParams1 = [featureOpt.corner_GOOD2TRACK, dict(maxCorners=100, qualityLevel=0.1, minDistance=15, blockSize=14)]
    featureParams2 = [featureOpt.corner_FAST, 30]
    cvNavImageProc = cvNaviInterface((300, 300), 1, cutOffbboxOpt1, featureParams1, featureParams2, 15)
    #-image process


    #+dataset Loading
    datasetPath = "dataset/"
    image_FileName, groundtruth_Filename, depth_Filename, IMULog, MotionContolLog, RoverCmdLog, depth_Width, depth_Height = getMetadata(datasetPath=datasetPath, filename="meta.json")
    RoverCmd = getTextData(RoverCmdLog)
    McLog = getTextData(MotionContolLog)
    #-dataset Loading


    #+ Robot Cmd Log Analysis
    cmdPeriod = getPeriodfromRobotCmd(RoverCmd, 60, len(McLog[0]))
    for period_idx, period in enumerate(cmdPeriod):
        for mcIdx in range(period.startIdx, period.endIdx):
            if McLog[1][mcIdx] != 0: #0:Stop, 1:Move, 2:Rotation
                if cmdPeriod[period_idx].ActualMvIdx == -1:
                    cmdPeriod[period_idx].ActualMvIdx = mcIdx
                if cmdPeriod[period_idx].ActualStopIdx < mcIdx:
                    cmdPeriod[period_idx].ActualStopIdx = mcIdx
        period.show()
    #- Robot Cmd Log Analysis

    # +ORB
    Orb3Trajectory = getTextData("dataset/ORB3_CameraTrajectory.txt")
    Orb_X = Orb3Trajectory[1] / 5
    Orb_Y = Orb3Trajectory[3] / 5
    # -ORB

    # +Motion Control Log Analysis
    odometry_Yaw = 0
    enc_odometry_X = [0]
    enc_odometry_Y = [0]
    ONEMETER_ENC = 4646
    NINTYDEG_ENC = 1000#966

    old_enc1 = 0
    old_enc2 = 0
    # -Motion Control Log Analysis

    if image_FileName != None:
        print(image_FileName, groundtruth_Filename, depth_Filename, depth_Width, depth_Height)

        cmdIdx = 0
        cmdIdxLim = 2#len(cmdPeriod)
        cap = cv2.VideoCapture(image_FileName)
        imgcnt = 0
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("No Cam")
                break

            #+image process
            #image = cv2.resize(image, (300, 300), interpolation=cv2.INTER_AREA)
            showOnceParam, longtermResult, shorttermResult = cvNavImageProc.getContextImg(image)
            showOnce(showOnceParam)
            #cv2.imshow('image', image)
            #-image process

            if imgcnt < cmdPeriod[cmdIdx].ActualMvIdx:
                imgcnt += 1
                continue

            if cmdPeriod[cmdIdx].ActualStopIdx < imgcnt:
                if cmdIdx < cmdIdxLim-1:
                    cmdIdx += 1

                    #+initilize period : 각 구간별 초기화 구문
                    cvNavImageProc.logntermInit() #image process
                    print("init:", imgcnt)
                    #+init encoder odometry
                    old_enc1 = 0
                    old_enc2 = 0
                    #-init encoder odometry

                    #-initilize period
                elif imgcnt < cmdPeriod[cmdIdx].endIdx:
                    break

            #+calculate encoder odometyr
            if McLog[1][imgcnt] != 0: #0:Stop, 1:Move, 2:Rotation
                delta_enc1 = McLog[3][imgcnt] - old_enc1
                delta_enc2 = McLog[4][imgcnt] - old_enc2

                trans_delta = (delta_enc1 + delta_enc2) / 2
                diff_delta = delta_enc1 - delta_enc2

                odometry_Yaw -= diff_delta / NINTYDEG_ENC * 90 / 2

                ox = enc_odometry_X[-1]
                oy = enc_odometry_Y[-1]

                ox += math.sin(odometry_Yaw / 180 * math.pi) * (trans_delta / ONEMETER_ENC)
                oy += math.cos(odometry_Yaw / 180 * math.pi) * (trans_delta / ONEMETER_ENC)

                enc_odometry_X.append(ox)
                enc_odometry_Y.append(oy)

                old_enc1 = McLog[3][imgcnt]
                old_enc2 = McLog[4][imgcnt]
            #-calculate encoder odometyr

            Depth_Map = np.zeros((depth_Height, depth_Width), np.float32)
            Depth_Mask = np.zeros((depth_Height, depth_Width, 3), np.uint8)
            if 0 != LoadBinDepthMapPnt(depth_Filename, depth_Width, depth_Height, 0, 10000, Depth_Map.ctypes, Depth_Mask.ctypes, imgcnt):
                norm_Depth_Map = cv2.normalize(Depth_Map, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
                #cv2.imshow("norm_Depth_Map", norm_Depth_Map)

                if 0 < len(longtermResult):
                    longtermBase = longtermResult[0]
                    longtermCur = longtermResult[1]
                    longtermIdx = longtermResult[2]
                    longtermCnt = longtermResult[3]
                    longtermDist = longtermResult[4]
                    longtermAng = longtermResult[5]
                    longtermInit = longtermResult[6]

                    if cmdPeriod[cmdIdx].ActionType == "MV":
                        pass
                    if cmdPeriod[cmdIdx].ActionType == "RT":
                        pass

                    if longtermInit == True:
                        print("Long Term Init", imgcnt)

            imgcnt += 1
            if cv2.waitKey(1) & 0xFF == 27:
                break
        cap.release()



    #+after algorithm processing
    enc_odometry_X = np.array(enc_odometry_X)
    enc_odometry_Y = np.array(enc_odometry_Y)
    #-after algorithm processing



    #+ shoing Graph
    plt.title('Vehicle Trajectory')
    orb = plt.scatter(Orb_X, Orb_Y, s=1)
    ref = plt.scatter(enc_odometry_X, enc_odometry_Y, s=1)
    plt.legend((orb, ref),
               ('ORB3 RGBD', 'Reference'),
               scatterpoints=1,
               loc='lower right',
               ncol=1,
               fontsize=10)
    plt.show()
    #-shoing Graph

