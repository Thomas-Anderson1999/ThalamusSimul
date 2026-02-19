
import sys
import cv2
import os
import numpy as np
import time
import math
import json
import socket as sock

from ThalamusEngine.Interface import *
from matplotlib import pyplot as plt
from MCLib.mcInterface import *
from ThalamusDSloader import *

#+UI
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
#-UI

import sys
from PyQt5.QtWidgets import QApplication, QWidget


#+Thalamus Navigation Logic
import os, sys
tmp = os.path.dirname(os.path.abspath(__file__))
tmp += "/ThalamusNavigation"
sys.path.append(tmp)
try:
    from ThalamusNavigation.NavInterface import *
except:
    print("You need ThalamusNavigation submodule")
#-Thalamus Navigation Logic

#+vision logic
try:
    from ThalamusNavigation.ThalamusCV.ThalamusCV.cvContext import *
    from ThalamusNavigation.ThalamusCV.ThalamusCV.cvUtil import *
except:
    print("You need ThalamusCV submodule")
#-vision logic



#+Detection Simulation
class detBoundingBox:
    def __init__(self, x1, y1, x2, y2, detClass, pos3D, objID):
        self.bboxPosX1 = x1
        self.bboxPosY1 = y1
        self.bboxPosX2 = x2
        self.bboxPosY2 = y2
        self.bboxCtrX = int((x1 + x2) / 2)
        self.bboxCtrY = int((y1 + y2) / 2)
        self.bboxWidth = x2 - x1
        self.bboxHeight = y2 - y1
        self.bboxClass = detClass
        self.pos3D = pos3D
        self.objId = objID

class detMilestne:
    def __init__(self, pos3D, cls):
        self.pos3D = pos3D
        self.classID = cls

def searchClosePnt(milestoneList, pos, th):
    for k in range(len(milestoneList)):
        if math.sqrt((milestoneList[k].pos3D[0] - pos[0])**2 + (milestoneList[k].pos3D[1] - pos[1])**2 + (milestoneList[k].pos3D[2] - pos[2])**2) < th:
            return k
    return -1

def getDetecColor(classID):
    color = (0, 0, 0)
    if classID == 0:
        color = (255, 255, 255)
    if classID == 1:
        color = (255, 0, 0)
    if classID == 2:
        color = (0, 255, 0)
    if classID == 3:
        color = (255, 255, 0)
    if classID == 4:
        color = (255, 0, 255)
    return color

def getSimulDetection():
    MaxBoundBoxNum = 128
    BoundBox = np.zeros(MaxBoundBoxNum * 4, np.int32)
    BoundBoxNum = GetBoundBox(BoundBox.ctypes)
    #print("BBNum " + str(BoundBoxNum))

    SrcHeight = 720
    SrcWidth = 1280
    BoundBoxImg = np.zeros((SrcHeight, SrcWidth, 3), np.uint8)
    GetColorImage(BoundBoxImg.ctypes, SrcWidth, SrcHeight)

    detList = []
    for i in range(BoundBoxNum):
        x1 = BoundBox[4 * i + 0]
        x2 = BoundBox[4 * i + 1]
        y1 = BoundBox[4 * i + 2]
        y2 = BoundBox[4 * i + 3]
        #print("BoundBox : {0} {1} {2} {3}".format(x1, y1, x2, y2))

        ctrX = int((x1 + x2) / 2)
        ctrY = int((y1 + y2) / 2)

        dist, objID, _, _, _, _ = ReturnDistance(ctrX, ctrY, False)
        ojbName = GetObjName(objID)
        if ojbName.find("DET") != -1:
            tmpSplit = ojbName.split("_")

            if i != objID: # not Los in sight
                continue
            if not(0 <= x1 and x1 < SrcWidth and 0 <= y1 and y1 <= SrcHeight):
                continue
            if not(0 <= x1 and x1 < SrcWidth and 0 <= y1 and y1 <= SrcHeight):
                continue

            print(ctrX, ctrY, dist)
            pos3D = list(Pixelto3D(ctrX, ctrY, dist))
            detList.append(detBoundingBox(x1, y1, x2, y2, int(tmpSplit[1]), pos3D, objID))
            cv2.rectangle(BoundBoxImg, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=3, lineType=cv2.LINE_4, shift=0)
            Sz = 0.3
            cv2.putText(BoundBoxImg, str(objID) + " " + ojbName, (ctrX, ctrY), cv2.FONT_HERSHEY_SIMPLEX, Sz, (0, 0, 255), 1, cv2.LINE_AA)

    return BoundBoxImg, detList

# -Detection Simulation

#+Util
def find_objbyname(search_name):
    objlist = []
    obj_id = 0
    modelLen, modelList = getModelList(1024)
    for mdlIdx, modelName in enumerate(modelList):
        if not(modelName == "ACTION_PUSH" or modelName == "ACTION_POP"):
            if -1 != modelName.find(search_name):
                objlist.append(obj_id)
            obj_id += 1
    return objlist
#-Util
#+Collision
def GetCollision(objid, skip_obj=[], verbose=False):
    res_col_list = []
    ret, col_list, col_num = CollisionCheck(objid)
    col_cnt = 0
    for col_idx in range(col_num):
        if col_list[col_idx] != 0 and (not col_idx in skip_obj):
            if verbose:
                print(col_idx, ":", col_list[col_idx], end=" ")
            res_col_list.append(col_idx)
            col_cnt += 1
    if col_cnt != 0 and verbose:
        print()
    return res_col_list
#+Collision

#+shared memory
from multiprocessing import shared_memory
def get_shm(_name, size_x, size_y, size_z, type=np.uint8):
    sample_array = np.zeros((size_x, size_y, size_z), dtype=type)
    try:
        stream_shm = shared_memory.SharedMemory(name =_name,create=True, size=sample_array.nbytes)
    except FileExistsError:
        stream_shm = shared_memory.SharedMemory(name=_name, create=False, size=sample_array.nbytes)
    return stream_shm

def EncodeJson2SHM(_buffer, data):
    b = np.ndarray(1024, dtype=np.byte, buffer=_buffer)
    json_bytes = str.encode(json.dumps(data))

    for idx in range(len(json_bytes)):
        b[idx] = json_bytes[idx]
    b[len(json_bytes)] = 0

def DecodeJsonFromSHM(_buffer):
    temp_bytes = np.ndarray(1024, dtype=np.byte, buffer=_buffer).tobytes()  # .decode(encoding="utf-8")
    temp_str = ""
    for idx in range(1024):
        if temp_bytes[idx] == 0:
            break
        else:
            temp_str += str(chr(int(temp_bytes[idx])))

    if 0 < len(temp_str):
        parsed_cmd = json.loads(temp_str)

        temp_bytes = np.ndarray(1024, dtype=np.byte, buffer=_buffer)
        for i in range(1024):
            temp_bytes[i] = 0
        return parsed_cmd
    return None


import threading
def RemoteSimulThread():
    remote_cnt = 0
    remote_rgb_img = get_shm("Thalamus_Simul_RGB", 1280, 720, 3)

    def getExtEngineImage(Color_width=1280, Color_Height=720, update=True):
        Color_image = np.zeros((Color_Height, Color_width, 3), np.uint8)
        if update:
            InitializeRenderFacet(-1, -1)  # refresh
        GetColorImage(Color_image.ctypes, Color_width, Color_Height)
        return Color_image

    while True:
        if 10 < remote_cnt:
            Remote_Img = getExtEngineImage(update=True)
            #cv2.imshow("Remote_Img", Remote_Img)
            #print("remote Thread")
            shared_a = np.ndarray(Remote_Img.shape, dtype=Remote_Img.dtype, buffer=remote_rgb_img.buf)
            shared_a[:] = Remote_Img

        time.sleep(0.050) #20hz
        remote_cnt += 1
#-shared memory



class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        mainGrid = QGridLayout()

        label = [["ScriptFile:", "EngineName:"]]
        editDefault = [["ScriptRover_LectureRoom.txt", "Thalamus QT Example"]]
        buttonText = ["Engine Start"]
        buttonFunc = [self.InitEngine]
        subgrid, self.startEdit = self.createGroupBox("Global Coord Test", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 0, 0)

        label = [["PosX", "PosY", "PosZ", "AttX", "AttY", "AttZ"]]
        editDefault = [["0", "0", "0", "0", "0", "0"]]
        buttonText = ["Get Global", "Pos Set", "Att Set", "Pos/Att Set", "BirdView"]
        buttonFunc = [self.getGlobal, self.globalPosSet, self.globalAttSet, self.globalPosAttSet, self.globalBirdview]
        subgrid, self.globalCoordEdit = self.createGroupBox("Global Coord Test", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 1, 0)

        label = [["ObjID", "TypeID", "ObjNAme"],["PosX", "PosY", "PosZ", "AttX", "AttY", "AttZ"], ["ClrX", "ClrY", "ClrZ", "AmpX", "AmpY", "AmpZ"]]
        editDefault = [["-1", "0", ""], ["0", "0", "0", "0", "0", "0"], ["0", "0", "0", "0", "0", "0"]]
        buttonText = ["objGetParam", "Type Set", "Name Set", "Pos Set", "Att Set", "Clr Set", "Amp Set", "Whole Set"]
        buttonFunc = [self.objGetParam,self.objSetType,self.objSetName,self.objSetPos,self.objSetAtt,self.objSetClr,self.objSetAmp,self.objSetParam]
        subgrid, self.objControlEdit = self.createGroupBox("Object Control", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 2, 0)

        label = [["ObjID", "ModelID", "Offset", "PosX", "PosY", "PosZ", "AttX", "AttY", "AttZ"]]
        editDefault = [["-1", "0", "-1", "0", "0", "0", "0", "0", "0"]]
        buttonText = ["MoelGetParam", "Param Set"]
        buttonFunc = [self.mdlGetParam, self.mdlSetParam]
        subgrid, self.mdlControlEdit = self.createGroupBox("Modeiling Control", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 3, 0)

        label = [["SrcPosX", "SrcPosY", "SrcWidth", "SrcHeight", "DestWidth", "DestHeight"], ["ObjID", "CPU Core", "Dataset"]]
        editDefault = [["0", "0", "1280", "720", "300", "300"], ["-1", "12", "01"]]
        buttonText = ["DepthMap", "ColorMap", "NoShade", "LightEff", "Bounding Box", "ext EngColor", "getRasterImg", "DataSet Adding"]
        buttonFunc = [self.funcDepthMap, self.funcColorMap, self.funcNoShade, self.funcLightEffect, self.funcBBox,
                      self.funcExtEngineViewMap, self.getRasterImg, self.funcDatasetAdding]
        subgrid, self.func1Edit = self.createGroupBox("Scene Generation", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 4, 0)

        label = [["DepthMap:", "Width", "Height", "MeshUp Inv", "FreeModelNum", "Thread"],["ColorImg:"]]
        editDefault = [["depthmap.txt","300", "300", "9", "5", "12"],["Dataset03/Color03.png"]]
        buttonText = ["MeshUp", "Texture Overay", "Texure Int", "TextureView", "pseudo lidar"]
        buttonFunc = [self.func2MeshUp, self.func2TexOveray, self.func2TexInt, self.func2TexView, self.Func2PsedoLidar]
        subgrid, self.func2Edit = self.createGroupBox("Mesh up, Texture Overay", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 5, 0)


        label = [["axis", "acc", "cruiseSpd", "dcc", "distance", "Simul Len"], ["Straight", "Rotate", "Cam Tilt", "File Save", "Vehicle"]]
        editDefault = [["0", "1.0", "1.0", "1.0", "3.0", "5.0"], ["1.0",  "45.0", "0", "None", "Rover1"]]
        buttonText = ["getProfile", "go Straigt", "go LR", "go Updn", "Rotate",
                      "cam Tilt", "View Sec1", "View Sec2", "View Sec3", "View Cam",
                      "Clicked WayPnt", "go Imd", "rot Imd", "tilt Imd"]
        buttonFunc = [self.motion_getProfile, self.motion_goFoward, self.motion_goLeftRight, self.motion_goUpdown, self.motion_Rotate,
                      self.motion_CamTilt, self.view_sec1, self.view_sec2, self.view_sec3, self.view_onCam,
                      self.clickedWayPnt, self.goImidiatly, self.rotImidiatly, self.tiltImidiatly]
        subgrid, self.func3Edit = self.createGroupBox("Motion Control", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 6, 0)

        label = [["1meter/pix", "moveable area", "GndPosX", "GndPosY", "TagPixX", "TagPixY"]]
        editDefault = [["50"," 25", "29", "31", "105", "206"]]
        buttonText = ["greedy Nav", "Navi Action", "Nav Clear", "MergeMap", "Test moving Obs", "Global Nav",
                      "Pred go St", "Pred rotate", "Detect Local"]
        buttonFunc = [self.greedyNav, self.naviAction, self.navClear, self.mergeMap, self.testMvObs, self.globalNavi,
                      self.pedictionGoStraight, self.pedictionRotate, self.DetectLocal]
        subgrid, self.func4Edit = self.createGroupBox("Thalamus Navigation Test", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 7, 0)

        self.setLayout(mainGrid)
        self.setWindowTitle("Thalamus Engine UI")

        self.resize(600, -1)

        #+Init Motion Controller
        InitMCSimul()
        axisIdx = InitContoller(1.0)
        #-Init Motion Controller

        # +Motion Constant
        self.timeSlice = GetTimeSlice()
        self.wheelCircum = math.pi * (0.22 + 0.03)
        # -Motion Constant

        #+Flag For Application
        self.vehicleCamMode = "NONE"
        #-Flag For Application

        #+Dataset setup
        self.datasetIndex = None
        #-Dataset setup

        #+Navagation value
        self.greedNavRes = None
        self.floorImgList = []
        self.floorCtrList = []
        self.rotposList = []
        self.initBasePosAtt = None
        self.milestoneList = []
        #-Navagation value

        #+moving obstacle
        self.movingObsParam = None
        #-moving obstacle

        #dataset meta file
        self.datasetMeta = {}
        self.datasetPath = "tmpDataset/"
        self.datasetMeta["depthmapFile"] = "DepthMap.bin"
        self.datasetMeta["image"] = "img.avi"
        self.datasetMeta["groundtruth"] = "groundTruth.txt"
        self.datasetMeta["DepthWidth"] = 0
        self.datasetMeta["DepthHeight"] = 0
        # dataset meta file

        #+udp server to remote contro;
        t1 = threading.Thread(target=self.udp_server_thread)
        t1.start()
        self.simulCnt = 0
        self.keepgoing_cnt = 0
        self.simulCnt = 0
        self.stoping_cnt = 0
        #-udp server to remote contro;

    def createGroupBox(self, gbxName, labeltext, editDefault, buttonText, buttonFunc):
        groupBox = QGroupBox(gbxName)
        grid = QGridLayout()
        groupBox.setLayout(grid)

        editList = []
        for j in range(len(labeltext)):
            k = 0
            for i in range(len(labeltext[j])):
                label = QLabel(self)
                label.setText(labeltext[j][i])
                grid.addWidget(label, j, k)
                k += 1

                edit = QLineEdit(self)
                edit.setText(editDefault[j][i])
                grid.addWidget(edit, j, k)
                editList.append(edit)
                k += 1

        for i in range(len(buttonText)):
            btn = QPushButton(buttonText[i], self)
            btn.clicked.connect(buttonFunc[i])
            grid.addWidget(btn, len(labeltext), i)

        return groupBox, editList

    def InitEngine(self):
        AsmFileName = self.startEdit[0].text().encode('UTF-8')
        SimWindowText = self.startEdit[1].text().encode('UTF-8')
        if True == LoadThalamusInterface():
            errCode = InitEngine(AsmFileName)
            if errCode != 0:
                errMsg = ""
                if errCode & 1 != 0:
                    errMsg += "env.txt "
                if errCode & 2 != 0:
                    errMsg += "script"
                QMessageBox.about(self, "Initialize Error:" + str(errCode), "Error On " + errMsg)

            StartExt3DEngine(AsmFileName, SimWindowText)

            self.Remote_timer = QTimer(self)
            self.Remote_timer.start(100)# 100ms
            self.Remote_timer.timeout.connect(self.RemoteSimulTimer)
            self.reverse_cmd = []
            """
            self.remote_cnt = 0
            self.remote_rgb_img = get_shm("Thalamus_Simul_RGB",1280,720, 3)
            """

            #+remote simulation
            self.Comm_Sim2Con = get_shm("Thalamus_Comm_Sim2Con", 1024, 1, 1)
            self.Comm_Con2Sim = get_shm("Thalamus_Comm_Con2Sim", 1024, 1, 1)
            self.CameraPosAtt = [0,0,0,0]
            t = threading.Thread(target=RemoteSimulThread)
            t.start()
            #-remote simulation
        else:
            OnMsgText("Error on Loading Library")

    #+ remote control simulation, 10Hz
    def RemoteSimulTimer(self):

        parsed_cmd = DecodeJsonFromSHM(self.Comm_Con2Sim.buf)

        if parsed_cmd is not None:

            self.func3Edit[10].setText(parsed_cmd["vehicle"])

            if parsed_cmd["cmd"] == "fwd":
                self.func3Edit[6].setText(str(parsed_cmd["param"]))
                self.motion_goFoward()

            if parsed_cmd["cmd"] == "move_LR":
                self.func3Edit[6].setText(str(parsed_cmd["param"]))
                self.motion_goLeftRight()

            if parsed_cmd["cmd"] == "updown":
                self.func3Edit[6].setText(str(parsed_cmd["param"]))
                self.motion_goUpdown()

            if parsed_cmd["cmd"] == "rotate":
                self.func3Edit[7].setText(str(parsed_cmd["param"]))
                self.motion_Rotate()


        data = {"battery": 100, "controll_height": 0, "sensing_height": int(self.CameraPosAtt[0] / 10),
                "attitude": {"roll":int(self.CameraPosAtt[1]), "pitch":int(self.CameraPosAtt[2]), "yaw":int(self.CameraPosAtt[3])}}
        EncodeJson2SHM(self.Comm_Sim2Con.buf, data)


        for cmd in self.reverse_cmd:
            cmd()
        self.reverse_cmd = []
    #udp server socket

    def udp_server_thread(self):
        server_ip = "127.0.0.1"
        server_port = 7726
        server_addr_port = (server_ip, server_port)
        buffersize = 1024

        udp_server_socket = sock.socket(family=sock.AF_INET, type=sock.SOCK_DGRAM)
        udp_server_socket.bind(server_addr_port)
        udp_server_socket.setblocking(False)
        # udp_server_socket.settimeout(1.0)

        print("UDP server is up and listening")

        # Listen Datagram incoming
        while (True):
            try:
                byte_addr_pair = udp_server_socket.recvfrom(buffersize)
            except BlockingIOError:
                continue

            """
            msg = byte_addr_pair[0]
            addr = byte_addr_pair[1]

            client_msg = "msg from client : {}".format(len(msg))
            client_ip = "client IP Addr : {}".format(addr)

            print(client_msg)
            print(client_ip)
            print(msg)
            """
            msg = byte_addr_pair[0].decode('utf-8')
            print(msg)

            parse = msg.split(' ')
            #SV -1 500                     XX
            if parse[0] == "SV":
                try:
                    self.Simultimer.stop()
                except:
                    pass
                axis = int(parse[1])
                param1 = int(parse[2])
                degree = 45 * (param1 - 500) / 500
                print(f"Servo {axis} {degree}")
                self.func3Edit[8].setText(str(degree))
                self.tiltImidiatly()
            #MV 1 1600 -4646               XX
            if parse[0] == "MV":
                try:
                    self.Simultimer.stop()
                except:
                    pass
                axis = int(parse[1])
                param1 = int(parse[2])
                param2 = int(parse[3])
                distance = -param2 / 4646
                print(f"Servo {axis} {distance}")
                self.func3Edit[6].setText(str(distance))
                self.reverse_cmd.append(self.motion_goFoward)
            #RT 1 800 4646                 XX
            if parse[0] == "RT":
                try:
                    self.Simultimer.stop()
                except:
                    pass
                axis = int(parse[1])
                param1 = int(parse[2])
                param2 = int(parse[3])
                angle = param2 / 4646 * 90
                print(f"Servo {axis} {angle}")
                self.func3Edit[7].setText(str(angle))
                self.reverse_cmd.append(self.motion_Rotate)
            if parse[0] == "KG":
                if 0 < self.keepgoing_cnt < len(self.leftPos) and 0 < self.simulCnt < len(self.leftPos):
                    self.left_kg_offset += (self.leftPos[self.simulCnt] - self.leftPos[self.keepgoing_cnt])
                    self.right_kg_offset += (self.rightPos[self.simulCnt] - self.rightPos[self.keepgoing_cnt])
                    self.simulCnt = self.keepgoing_cnt
                print("Keep Going", self.keepgoing_cnt, self.simulCnt)
            if parse[0] == "ST":
                if 0 <= self.stoping_cnt < len(self.leftPos) and 0 <= self.simulCnt < len(self.leftPos):
                    self.left_kg_offset -= (self.leftPos[self.stoping_cnt] - self.leftPos[self.simulCnt])
                    self.right_kg_offset -= (self.rightPos[self.stoping_cnt] - self.rightPos[self.simulCnt])
                    self.simulCnt = self.stoping_cnt
                print("Stoping", self.stoping_cnt, self.simulCnt)
    # - remote control simulation

    #+Global Coord Button Func
    def getGlobal(self):
        px, py, pz = GetGlobalPos()
        ax, ay, az = GetGlobalAtt()
        self.globalCoordEdit[0].setText(str(px))
        self.globalCoordEdit[1].setText(str(py))
        self.globalCoordEdit[2].setText(str(pz))
        self.globalCoordEdit[3].setText(str(ax))
        self.globalCoordEdit[4].setText(str(ay))
        self.globalCoordEdit[5].setText(str(az))

    def globalPosSet(self):
        x = float(self.globalCoordEdit[0].text())
        y = float(self.globalCoordEdit[1].text())
        z = float(self.globalCoordEdit[2].text())
        #SetGlobalPosition(x,y,z)
        setModelPosRot(1, x, y, z, 0, 0, 0)
        InitializeRenderFacet(-1, -1)
    def globalAttSet(self):
        x = float(self.globalCoordEdit[3].text())
        y = float(self.globalCoordEdit[4].text())
        z = float(self.globalCoordEdit[5].text())
        #SetGlobalAttitude(x, y, z)
        setModelPosRot(0, 0, 0, 0, x, y, z)
        InitializeRenderFacet(-1, -1)
    def globalPosAttSet(self):
        self.globalPosSet()
        self.globalAttSet()
    def globalBirdview(self):
        self.globalCoordEdit[1].setText("8000")
        self.globalCoordEdit[2].setText("0")
        self.globalCoordEdit[3].setText("90")
        self.globalPosAttSet()
        self.vehicleCamMode = "NONE"

    #-Global Coord Button Func

    #+Obj Control
    def objGetParam(self):
        objID = GetHighLightedObj()
        if -1 != objID:
            self.objControlEdit[0].setText(str(objID))
            self.objControlEdit[1].setText(str(GetObjType(objID)))
            self.objControlEdit[2].setText(GetObjName(objID))
            x,y,z = GetObjPos(objID)
            self.objControlEdit[3].setText(str(x))
            self.objControlEdit[4].setText(str(y))
            self.objControlEdit[5].setText(str(z))
            x, y, z = GetObjAtt(objID)
            self.objControlEdit[6].setText(str(x))
            self.objControlEdit[7].setText(str(y))
            self.objControlEdit[8].setText(str(z))
            x, y, z = GetObjClr(objID)
            self.objControlEdit[9].setText(str(x))
            self.objControlEdit[10].setText(str(y))
            self.objControlEdit[11].setText(str(z))
            x, y, z = GetObjAmp(objID)
            self.objControlEdit[12].setText(str(x))
            self.objControlEdit[13].setText(str(y))
            self.objControlEdit[14].setText(str(z))
    def objSetType(self):
        objID = int(self.objControlEdit[0].text())
        if objID == -1:
            type = int(self.objControlEdit[1].text())
            SetObjectType(objID, type)
            InitializeRenderFacet(-1, -1)
            InitializeRenderFacet(-1, -1)
    def objSetName(self):
        print("sss")
        objID = int(self.objControlEdit[0].text())
        if objID != -1:
            SetObjName(objID, self.objControlEdit[2].text())
    def objSetPos(self):
        objID = int(self.objControlEdit[0].text())
        if objID != -1:
            x = float(self.objControlEdit[3].text())
            y = float(self.objControlEdit[4].text())
            z = float(self.objControlEdit[5].text())
            SetObjPos(objID, x, y, z)
            InitializeRenderFacet(-1, -1)
    def objSetAtt(self):
        objID = int(self.objControlEdit[0].text())
        if objID != -1:
            x = float(self.objControlEdit[6].text())
            y = float(self.objControlEdit[7].text())
            z = float(self.objControlEdit[8].text())
            SetObjAtt(objID, x, y, z)
            InitializeRenderFacet(-1, -1)
    def objSetClr(self):
        objID = int(self.objControlEdit[0].text())
        if objID == -1:
            x = float(self.objControlEdit[9].text())
            y = float(self.objControlEdit[10].text())
            z = float(self.objControlEdit[11].text())
            SetObjClr(objID, x, y, z)
            InitializeRenderFacet(-1, -1)
    def objSetAmp(self):
        objID = int(self.objControlEdit[0].text())
        if objID != -1:
            x = float(self.objControlEdit[12].text())
            y = float(self.objControlEdit[13].text())
            z = float(self.objControlEdit[14].text())
            SetObjAmp(objID, x, y, z)
            InitializeRenderFacet(-1, -1)
    def objSetParam(self):
        self.objSetType()
        self.objSetName()
        self.objSetPos()
        self.objSetAtt()
        self.objSetAmp()
    #-Obj Control

    #+Model Control
    def mdlGetParam(self):
        objID = GetHighLightedObj()
        self.mdlControlEdit[0].setText(str(objID))
        if objID != -1:
            ModelID = getModelIDByObjID(objID)
            if ModelID != -1:
                self.mdlControlEdit[1].setText(str(ModelID))
                Modleffset = int(self.mdlControlEdit[2].text())
                res, data = getModelPosRot(ModelID + Modleffset)
                if res != 0:
                    self.mdlControlEdit[3].setText(str(data[0]))
                    self.mdlControlEdit[4].setText(str(data[1]))
                    self.mdlControlEdit[5].setText(str(data[2]))
                    self.mdlControlEdit[6].setText(str(data[3]))
                    self.mdlControlEdit[7].setText(str(data[4]))
                    self.mdlControlEdit[8].setText(str(data[5]))


    def mdlSetParam(self):
        objID = GetHighLightedObj()
        self.mdlControlEdit[0].setText(str(objID))
        if objID != -1:
            ModelID = getModelIDByObjID(objID)
            if ModelID != -1:
                self.mdlControlEdit[1].setText(str(ModelID))
                Modleffset = int(self.mdlControlEdit[2].text())
                px = float(self.mdlControlEdit[3].text())
                py = float(self.mdlControlEdit[4].text())
                pz = float(self.mdlControlEdit[5].text())
                ax = float(self.mdlControlEdit[6].text())
                ay = float(self.mdlControlEdit[7].text())
                az = float(self.mdlControlEdit[8].text())
                setModelPosRot(ModelID + Modleffset, px, py, pz, ax, ay, az)
                InitializeRenderFacet(-1,-1)
    #-Model Control
    #+Function 1
    def getFunc1Param(self):
        SrcPosX = int(self.func1Edit[0].text())
        SrcPosY = int(self.func1Edit[1].text())
        SrcWidth = int(self.func1Edit[2].text())
        SrcHeight = int(self.func1Edit[3].text())
        DestWidth = int(self.func1Edit[4].text())
        DestHeight = int(self.func1Edit[5].text())
        ObjID = int(self.func1Edit[6].text())
        CPUCore = int(self.func1Edit[7].text())
        return SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore
    def funcDepthMap(self):
        print("depth map")

        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()

        Depth_Map = np.zeros((DestHeight, DestWidth), np.float32)
        Depth_Mask = np.zeros((DestHeight, DestWidth, 3), np.uint8)

        t0 = time.monotonic()
        GetDepthMap(Depth_Map.ctypes, Depth_Mask.ctypes, DestWidth, DestHeight, CPUCore, SrcPosX, SrcPosY, SrcWidth, SrcHeight, ObjID)
        t1 = time.monotonic() - t0
        print("Time elapsed: ", t1)

        SaveRawSeperateDepthFile('depthmap.txt', Depth_Map)

        ObjIDMask, FaceIDMask, EdgeMask = cv2.split(Depth_Mask)
        Depth_Map = cv2.normalize(Depth_Map, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        cv2.imshow("Depth Map", Depth_Map)
        cv2.imshow("Depth Mask", EdgeMask)
    def funcColorMap(self):
        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()

        Shade_Mask = np.zeros((DestHeight, DestWidth, 3), np.uint8)
        Shade_Img = np.zeros((DestHeight, DestWidth, 3), np.uint8)

        GetShadeImage(Shade_Img.ctypes, Shade_Mask.ctypes, DestWidth, DestHeight, CPUCore, SrcPosX, SrcPosY, SrcWidth, SrcHeight, ObjID)
        cv2.imshow("Shade_Img", Shade_Img)

    def getExtEngineImage(self, Color_width=1280, Color_Height=720, update=True):
        Color_image = np.zeros((Color_Height, Color_width, 3), np.uint8)
        if update:
            InitializeRenderFacet(-1, -1)  # refresh
        GetColorImage(Color_image.ctypes, Color_width, Color_Height)
        return Color_image
    def funcExtEngineViewMap(self):
        Color_image = self.getExtEngineImage()
        cv2.imshow("External Engine Color Image", Color_image)
        cv2.imwrite("extColor.png", Color_image)

    def getRasterImg(self):
        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()
        Color_width = 640
        Color_Height = 360

        Color_image = np.zeros((Color_Height, Color_width, 3), np.uint8)
        Depth_Map = np.zeros((Color_Height, Color_width), np.float32)
        Depth_Mask = np.zeros((Color_Height, Color_width, 3), np.uint8)

        InitializeRenderFacet(-1, -1)  # refresh

        GetRasterizedImage(Color_image.ctypes, Depth_Map.ctypes, Depth_Mask.ctypes,
                           Color_width, Color_Height, CPUCore, SrcPosX, SrcPosY, SrcWidth, SrcHeight, ObjID)
        cv2.imshow("Rasterizing Color Image", Color_image)

        ObjIDMask, FaceIDMask, EdgeMask = cv2.split(Depth_Mask)
        Depth_Map = cv2.normalize(Depth_Map, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
        cv2.imshow("Depth Map", Depth_Map)
        cv2.imshow("Depth Mask", EdgeMask)


    def funcDatasetAdding(self):
        roverBaseID = find_objbyname(self.VehicleName)[0]
        datasetPath = "dataset"
        datasetWH = 300

        #+Initialize Dataset
        if self.datasetIndex is None:
            self.datasetIndex = 0
            try:
                if not os.path.exists(datasetPath):
                    os.makedirs(datasetPath)
            except OSError:
                print("Error: Failed to create the dataset directory. " + datasetPath)

            datasetName = self.func1Edit[8].text()
            datasetPath = datasetPath + "/" + datasetName
            try:
                if not os.path.exists(datasetPath):
                    os.makedirs(datasetPath)
            except OSError:
                print("Error: Failed to create the dataset directory. " + datasetPath)

            with open(datasetPath + "/dsLog.txt", "w") as f:
                pass

            _, _, self.offsetDSPosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
        else:
            datasetName = self.func1Edit[8].text()
            datasetPath = datasetPath + "/" + datasetName
        # -Initialize Dataset


        with open(datasetPath + "/dsLog.txt", "a+") as f:
            posModelIO, rotModelIO, basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
            datasetAdd = np.array(basePosAtt) - np.array(self.offsetDSPosAtt)
            print(datasetAdd)
            f.write(str(self.datasetIndex) + " " + str(datasetAdd[0]) + " " + str(datasetAdd[1]) + " " + str(datasetAdd[2])
                    + " " + str(datasetAdd[3]) + " " + str(datasetAdd[4]) + " " + str(datasetAdd[5])+"\n")

        #+Color Img
        Color_width = 1280
        Color_Height = 720
        Color_image = np.zeros((Color_Height, Color_width, 3), np.uint8)
        InitializeRenderFacet(-1, -1)  # refresh
        GetColorImage(Color_image.ctypes, Color_width, Color_Height)
        Color_image = cv2.resize(Color_image, (datasetWH, datasetWH), cv2.INTER_LANCZOS4)
        #cv2.imshow("External Engine Color Image", Color_image)
        fname = datasetPath + "/colpr" + str("%03d" % self.datasetIndex)+".png"
        cv2.imwrite(fname, Color_image)
        #-Color Img


        #+depth map
        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()
        Depth_Map = np.zeros((DestHeight, DestWidth), np.float32)
        Depth_Mask = np.zeros((DestHeight, DestWidth, 3), np.uint8)
        t0 = time.monotonic()
        GetDepthMap(Depth_Map.ctypes, Depth_Mask.ctypes, DestWidth, DestHeight, CPUCore, SrcPosX, SrcPosY, SrcWidth, SrcHeight, ObjID)
        t1 = time.monotonic() - t0
        print("Time elapsed: ", t1)
        fname = datasetPath + "/depth" + str("%03d" % self.datasetIndex) + ".txt"
        SaveRawSeperateDepthFile(fname, Depth_Map)
        #-depth map

        self.datasetIndex += 1

    def funcNoShade(self):
        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()

        Shade_Mask = np.zeros((DestHeight, DestWidth, 3), np.uint8)
        Shade_Img = np.zeros((DestHeight, DestWidth, 3), np.uint8)

        GetColorImageNoShade(Shade_Img.ctypes, Shade_Mask.ctypes, DestWidth, DestHeight, CPUCore, SrcPosX, SrcPosY, SrcWidth, SrcHeight)
        cv2.imshow("Shade_Img", Shade_Img)
        cv2.imwrite("Shade_Img.png", Shade_Img)
    def funcLightEffect(self):
        print("sss")

    def funcBBox(self):
        print("Bound Box")
        MaxBoundBoxNum = 128
        BoundBox = np.zeros(MaxBoundBoxNum * 4, np.int32)
        BoundBoxNum = GetBoundBox(BoundBox.ctypes)
        print("BBNum " + str(BoundBoxNum))
        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()

        BoundBoxImg = np.zeros((SrcHeight,SrcWidth, 3), np.uint8)
        GetColorImage(BoundBoxImg.ctypes, SrcWidth, SrcHeight)
        for i in range(BoundBoxNum):
            x1 = BoundBox[4 * i + 0]
            x2 = BoundBox[4 * i + 1]
            y1 = BoundBox[4 * i + 2]
            y2 = BoundBox[4 * i + 3]
            print("BoundBox : {0} {1} {2} {3}".format(x1, y1, x2, y2))
            cv2.rectangle(BoundBoxImg, pt1=(x1, y1), pt2=(x2, y2), color=(0, 255, 0), thickness=1, lineType=cv2.LINE_4,
                          shift=0)
        cv2.imshow("BoundBox", BoundBoxImg)
    # -Function 1

    #+Function 2
    #buttonFunc = [self.func2MeshUp, self.func2TexOveray, self.func2TexInt, self.func2TexView]

    def Func2PsedoLidar(self):

        roverBaseID = find_objbyname(self.VehicleName)[0]
        posModelIO, rotModelIO, basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation

        px = basePosAtt[0]
        pz = basePosAtt[2]
        pyaw = basePosAtt[4]

        px = 0
        pz = 3000
        pyaw = 180

        sampleY = int(720/2)



        # + actual sensor sampling
        """
        InitializeRenderFacet(-1, -1)
        r, _, _, _, _, _ = ReturnDistance(0, 0, True)
        actual_Pos = []
        for x in range(1280):
            r, _, _, _, _, _ = ReturnDistance(int(x), sampleY, False)
            if r < 6000:
                pos = Pixelto3D(x, sampleY, r)
                pos = list(pos)

                pos[0] *= -1
                pos[2] *= -1

                pos[0] += px
                pos[2] += pz
                actual_Pos.append(pos)
        actual_Pos = np.array(actual_Pos)
        """
        # - actual sensor sampling

        actual_Pos = getTextData("temp_recon.txt")
        actual_Pos[0] *= -1
        actual_Pos[2] *= -1
        actual_Pos[2] += 1100

        #+ reconstruction sampling
        SetProcessingEngineIndex(1)
        InitializeRenderFacet(-1, -1)
        r, _, _, _, _, _ = ReturnDistance(0, 0, True)

        recon_Pos = []
        #f = open("temp_recon.txt", 'w')
        for x in range(1280):
            r, _, _, _, _, _ = ReturnDistance(int(x), sampleY, False)
            if r < 6000:
                pos = Pixelto3D(x, sampleY, r)
                recon_Pos.append([pos[0], pos[1], pos[2]])
                #f.write(str(pos[0]) + " " + str(pos[1]) + " " + str(pos[2]) + "\r\n")
        #f.close()
        SetProcessingEngineIndex(0)
        recon_Pos = np.array(recon_Pos)
        # - reconstruction sampling


        plt.title('Pseudo Sensor')
        #actual = plt.scatter(actual_Pos[:, 0], actual_Pos[:, 2], s=1)
        actual = plt.scatter(actual_Pos[0], actual_Pos[2], s=1)
        recon = plt.scatter(recon_Pos[:,0], recon_Pos[:,2], s=1)

        plt.legend((actual, recon),
                   ('Current FOV', 'Reconstruction Sampling'),
                   scatterpoints=1,
                   loc='upper left',
                   ncol=1,
                   fontsize=10)
        plt.arrow(px, pz, 0, -1, head_width=0.05, head_length=0.05, fc='red', ec='red')
        plt.axis('equal')

        plt.show()

        pass

    def func2MeshUp(self):

        depWidth = int(self.func2Edit[1].text())
        depHeight = int(self.func2Edit[2].text())
        depInv = int(self.func2Edit[3].text())
        MeshUpType = int(self.func2Edit[4].text())

        Depth_Map = np.zeros((depHeight, depWidth), np.float32)
        Depth_Mask = np.zeros((depHeight, depWidth, 3), np.uint8)

        AsmFileName = b"ScriptFreeModel.txt"
        InitEngine(AsmFileName, 1280, 720, 1)
        SetProcessingEngineIndex(1)
        if 0 != LoadBinDepthMapPnt(self.func2Edit[0].text(), depWidth, depHeight, 600, 6000, Depth_Map.ctypes, Depth_Mask.ctypes):
            ret = ObjMeshUp(depWidth, depHeight, MeshUpType, depInv)

            Depth_Map = cv2.normalize(Depth_Map, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
            cv2.imshow("Depth Map", Depth_Map)

            print(ret)
            InitializeRenderFacet(-1, -1)
        else:
            print("Loading Error")
        SetProcessingEngineIndex(0)

    def func2TexOveray(self):
        depWidth = int(self.func2Edit[1].text())
        depHeight = int(self.func2Edit[2].text())
        TheadNum = int(self.func2Edit[5].text())
        Texture = cv2.imread(self.func2Edit[6].text())
        Texture = cv2.resize(Texture, (depWidth, depHeight))
        cv2.imshow("Texture Src", Texture)

        _, _, SrcWidth, SrcHeight, _, _, _, _ = self.getFunc1Param()
        TexureOveray(TheadNum, Texture.ctypes, depWidth, depHeight, SrcWidth, SrcHeight)
    def func2TexInt(self):
        TheadNum = int(self.func2Edit[5].text())
        TextureInterpolation(TheadNum)
        print("Interpolation")

    def func2TexView(self):
        TheadNum = int(self.func2Edit[5].text())
        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()
        TextuedView = np.zeros((DestHeight, DestWidth, 3), np.uint8)

        getTextureImg(TheadNum, TextuedView.ctypes, SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID)
        cv2.imshow("Texture View", TextuedView)
    #-Function 2

    #+Function Motion Control
    def motion_getProfile(self):
        #+ Profile Parameter
        axisIdx = int(self.func3Edit[0].text())
        acc = float(self.func3Edit[1].text())
        cruiseSpd = float(self.func3Edit[2].text())
        dcc = float(self.func3Edit[3].text())
        distance = float(self.func3Edit[4].text())
        simulTime = float(self.func3Edit[5].text())
        # - Profile Parameter

        print("axisIdx:", axisIdx)
        print(MoveDS(axisIdx, acc, cruiseSpd, dcc, distance))  # acc, cruise_spd, dcc, distance
        ret, Time, Spd, Pos, State = getProfile(axisIdx, simulTime)


        plt.plot(Time, Spd, label="Speed Profile")
        plt.ylabel('Speed of profile(m/sec)', fontsize=12)
        plt.xlabel('Time(ms)', fontsize=12)
        #plt.plot(Time, Pos, label="Position Profile")
        plt.legend()
        plt.show()
        """
        plt.plot(Time, Pos, label="distance")
        plt.legend()
        plt.show()
        """

    def motionInit(self, leftSign, rightSign, distance, inputSpd=None):
        # + Profile Parameter
        axisIdx = int(self.func3Edit[0].text())
        acc = float(self.func3Edit[1].text())
        if inputSpd is None:
            cruiseSpd = float(self.func3Edit[2].text())
        else:
            cruiseSpd = inputSpd
        dcc = float(self.func3Edit[3].text())
        # distance = float(self.func3Edit[4].text())
        simulTime = float(self.func3Edit[5].text())
        # - Profile Parameter

        # +get profile
        print(MoveDS(axisIdx, acc, cruiseSpd, dcc, distance))  # acc, cruise_spd, dcc, distance
        ret, Time, Spd, Pos, State = getProfile(axisIdx, simulTime)
        self.leftPos = leftSign * Pos.copy()
        self.rightPos = rightSign * Pos.copy()
        self.leftSpd = leftSign * Spd.copy()
        self.rightSpd = rightSign * Spd.copy()

        #+ smoothing torque : for drone motion
        len_torque = len(self.rightSpd)
        for idx in range(len(State)):
            if State[idx] == 3:
                len_torque = idx
                break
        self.smooth_torque = np.zeros(len_torque, np.float32)
        mv_bufflen = 500
        mv_idx = 0
        for _idx in range(len_torque//2):
            idx = _idx
            if _idx == 0:
                mv_buff = np.zeros(mv_bufflen, np.float32)
            else:
                mv_buff[mv_idx % mv_bufflen] = self.rightSpd[idx] - self.rightSpd[idx-1]
            self.smooth_torque[idx] = np.average(mv_buff)
            mv_idx += 1
        for _idx in range(len_torque-1, len_torque//2-1, -1):
            idx = _idx
            if _idx == len_torque-1:
                mv_buff = np.zeros(mv_bufflen, np.float32)
            else:
                mv_buff[mv_idx % mv_bufflen] = self.rightSpd[idx] - self.rightSpd[idx-1]
            self.smooth_torque[idx] = np.average(mv_buff)
            mv_idx += 1
        # - smoothing torque
        """
        plt.plot(self.smooth_torque, label="smooth_torque")
        plt.legend()
        plt.show()
        """
        self.motionState = State
        # -get profile

    def startSimulTimer(self, timerFunc):
        # +start Simul Timer
        self.VehicleName = self.func3Edit[10].text()
        self.DroneMode = False
        if self.VehicleName.lower().find("drone") != -1:
            self.DroneMode = True
        self.Simultimer = QTimer(self)
        self.Simultimer.start(self.timeSlice)
        self.Simultimer.timeout.connect(timerFunc)
        self.simulCnt = 0
        self.refreshRate = 12
        # -start Simul Timer

    def getSrcPosAtt(self, baseID, posOffset, rotOffset):
        posModelIO = getModelIDByObjID(baseID) + posOffset
        ret, Buff = getModelPosRot(posModelIO)
        src_px = Buff[0]
        src_py = Buff[1]
        src_pz = Buff[2]
        rotModelIO = getModelIDByObjID(baseID) + rotOffset
        ret, Buff = getModelPosRot(rotModelIO)
        src_pitch = Buff[3]
        src_yaw = Buff[4]
        src_roll = Buff[5]
        # - get Src Position / Rotation
        basePosAtt = [src_px, src_py, src_pz, src_pitch, src_yaw, src_roll]
        return posModelIO, rotModelIO, basePosAtt

    def motion_goFoward(self):
        distance = float(self.func3Edit[6].text()) # +go Foward Distance

        #+drone mode param
        self.current_moving_distance = distance
        self.rotation_stat = False
        self.go_leftright = False
        self.go_updown = False
        #-drone mode param

        if 0 < distance:
            self.motionInit(1, 1, distance)
        else:
            self.motionInit(-1, -1, -distance)
        self.startSimulTimer(self.simulLocoTimer_slot)

    def motion_goLeftRight(self):
        distance = float(self.func3Edit[6].text()) # +go Foward Distance

        #+drone mode param
        self.current_moving_distance = distance
        self.rotation_stat = False
        self.go_leftright = True
        self.go_updown = False
        #-drone mode param

        if 0 < distance:
            self.motionInit(1, 1, distance)
        else:
            self.motionInit(-1, -1, -distance)
        self.startSimulTimer(self.simulLocoTimer_slot)

    def motion_goUpdown(self):

        distance = float(self.func3Edit[6].text()) # +go Foward Distance

        #+drone mode param
        self.current_moving_distance = distance
        self.rotation_stat = False
        self.go_leftright = False
        self.go_updown = True
        #-drone mode param

        if 0 < distance:
            self.motionInit(1, 1, distance)
        else:
            self.motionInit(-1, -1, -distance)
        self.startSimulTimer(self.simulLocoTimer_slot)

    def motion_Rotate(self):
        angle = float(self.func3Edit[7].text())
        distance = angle / 180. * self.wheelCircum #Unit:M

        # +drone mode param
        self.current_moving_distance = distance
        self.rotation_stat = True
        self.go_leftright = False
        self.go_updown = False
        # -drone mode param

        print("distance:", distance)
        if 0 < angle:
            self.motionInit(1, -1, distance)
        else:
            self.motionInit(-1, 1, -distance)
        self.startSimulTimer(self.simulLocoTimer_slot)

    def motion_CamTilt(self):
        camBaseID = 3
        self.posModelIO, self.rotModelIO, self.basePosAtt = self.getSrcPosAtt(camBaseID, -2,-1)  # get Src Position / Rotation

        angle = float(self.func3Edit[8].text())
        distance = angle - self.basePosAtt[3]

        if 0 < distance:
            self.motionInit(1, -1, distance, 20)
        else:
            self.motionInit(-1, 1, -distance, 20)
        self.startSimulTimer(self.simulCamTiltTimer_slot)

    def setCamView(self, Yaw, Pitch, Roll=0, Update=True):
        # +Initiaizize Pos/Att
        setModelPosRot(0, 0, 0, 0, 0, 0, 0)
        setModelPosRot(1, 0, 0, 0, 0, 0, 0)
        # -Initiaizize Pos/Att

        mdlFrame = getLocalFrame(self.camMdlIdx)
        CamSz = 30 # Cam depth(size Z) is 30, somwhat margin 15->30 casue of pitch movement
        CamOffsetY = 30 # camera position for offset

        camOffsetX = CamSz * math.sin(Yaw * math.pi / 180.)
        camOffsetZ = CamSz * math.cos(Yaw * math.pi / 180.) + CamOffsetY * math.sin(Pitch * math.pi / 180.)
        camOffsetY = CamOffsetY * math.cos(Pitch * math.pi / 180.)

        setModelPosRot(0, 0,0, 0, Pitch, 0, 0)
        setModelPosRot(1, -(mdlFrame[3][0] + camOffsetX), -mdlFrame[3][1]+camOffsetY, -(mdlFrame[3][2] + camOffsetZ), 0, -Yaw, 0)
        if Update:
            self.lastPitch = Pitch #for locomotion Timer's Initial Value
            self.lastYaw = Yaw #for locomotion Timer's Initial Value
    def simulLocoTimer_slot(self):
        if self.simulCnt == len(self.motionState): #state_standby
            self.Simultimer.stop()
            return

        #Initialize Movement
        if self.simulCnt == 0:
            #InitializeRenderFacet(-1, -1)
            # + 모션 컨트롤 state 분석, keep going할때와 stop시작할 카운트를 저장해 둔다
            for idx in range(len(self.motionState)):
                if self.motionState[idx] != 0:
                    self.keepgoing_cnt = idx
                    break
            for idx in range(len(self.motionState)):
                if self.motionState[idx] == 2:
                    self.stoping_cnt = idx
                    break
            self.left_st_offset = 0
            self.right_st_offset = 0
            self.left_kg_offset = 0
            self.right_kg_offset = 0
            # - 모션 컨트롤 state 분석, keep going할때와 stop시작할 카운트를 저장해 둔다

            distance = float(self.func3Edit[6].text())  # +go Foward Distance
            self.distance_sign = np.sign(distance)

            self.vehicle_obj = find_objbyname(self.VehicleName)
            roverBaseID = self.vehicle_obj[0]
            self.posModelIO, self.rotModelIO, self.basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3) #get Src Position / Rotation
            self.zDiff = 0
            self.xDiff = 0
            self.yDiff = 0

            #+ drone connecting process
            self.drone_move_together = False
            if -1 != self.VehicleName.lower().find("Drone"):
                self._vehicle_obj = find_objbyname("Drone1")
                droneBaseID = self._vehicle_obj[0]
                self._posModelIO, self._rotModelIO, self._basePosAtt = self.getSrcPosAtt(droneBaseID, -4,-3)  # get Src Position / Rotation
                collision_list = GetCollision(self._vehicle_obj[0], self._vehicle_obj)
                for objid in self.vehicle_obj:
                   if objid in collision_list:
                       self.drone_move_together = True
            #- drone connecting process


            #+video
            self.vidOut = None
            if self.func3Edit[9].text() != "None":
                try:
                    os.mkdir(self.datasetPath)
                except:
                    pass
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                self.datasetMeta["image"] = self.func3Edit[9].text() + '.avi'
                self.vidOut = cv2.VideoWriter(self.datasetPath + self.datasetMeta["image"], fourcc, 30, (1280, 720))

                #+metafile for dataset
                _, _, _, _, DestWidth, DestHeight, _, _ = self.getFunc1Param()
                self.datasetMeta["DepthWidth"] = DestWidth
                self.datasetMeta["DepthHeight"] = DestHeight
                with open(self.datasetPath +'meta.json', 'w') as fp:
                    json.dump(self.datasetMeta, fp)
                #-metafile for dataset

            #-video
        #+locomotion Modeling

        #+Calculation Yaw
        tmpDiff = (self.leftPos[self.simulCnt] + self.left_kg_offset - (self.rightPos[self.simulCnt] + self.right_kg_offset)) / 2
        rotateLoop = 2 * self.wheelCircum * 1000
        tmpSign = np.sign(tmpDiff)
        tmpDiff -= int(tmpDiff / rotateLoop) * rotateLoop * tmpSign
        Yaw = tmpDiff / rotateLoop * 360. + self.basePosAtt[4]
        # -Calculation Yaw

        tmpPosDiff = (self.leftSpd[self.simulCnt] + self.rightSpd[self.simulCnt]) / 2

        if self.go_leftright:  # drone mode rotation doesn't change yaw
            Yaw = self.basePosAtt[4]
            self.xDiff += tmpPosDiff * self.timeSlice * math.cos(Yaw * math.pi / 180.)
            self.zDiff += tmpPosDiff * self.timeSlice * math.sin(Yaw * math.pi / 180.)
            self.yDiff = 0
        elif self.go_updown:
            self.yDiff += tmpPosDiff * self.timeSlice
            self.xDiff = 0
            self.zDiff = 0
        else: #moving foward/backward
            self.xDiff += tmpPosDiff * self.timeSlice * math.sin(Yaw * math.pi / 180.)
            self.zDiff += tmpPosDiff * self.timeSlice * math.cos(Yaw * math.pi / 180.)
            self.yDiff = 0
        #-locomotion Modeling

        #+moving obstacle : 움직이는 장애물 Script에서 NAME OBS01
        if self.movingObsParam is not None:
            objID = 7
            obspos = list(GetObjPos(objID))
            if obspos[2] < self.movingObsParam[0]:
                obspos[2] += self.movingObsParam[1]
                SetObjPos(objID, obspos[0], obspos[1], obspos[2])
                #InitializeRenderFacet(-1, -1)
            else:
                self.movingObsParam = None
        #-moving obstacle

        #+Drone mode Locomotion
        locomotion_pitch = 0
        locomotion_roll = 0
        if self.DroneMode:
            if 0 < self.current_moving_distance:
                locomotion_pitch = -8 * self.smooth_torque[self.simulCnt] * 1000 * self.distance_sign
            else:
                locomotion_pitch = +8 * self.smooth_torque[self.simulCnt] * 1000 * self.distance_sign
            if self.rotation_stat or self.go_updown:
                locomotion_pitch = 0
            if self.go_leftright:
                locomotion_roll = -locomotion_pitch
                locomotion_pitch = 0
        # -Drone mode Locomotion

        # +collision check
        collision_list = GetCollision(self.vehicle_obj[0], self.vehicle_obj)
        if 0 < len(collision_list):
            print("collision:", collision_list, self.simulCnt)
            if 300 < self.simulCnt:
                self.Simultimer.stop()
        # -collision check

        # + get floor sensor data : ONLY vehicleCamMode == True
        floor_sensor = ReturnDistanceByPos2Dir(0, 20, 0, 0, 1, 0) #
        #print("floor_sensor:", floor_sensor, self.basePosAtt[0] + self.xDiff, self.basePosAtt[1] + self.yDiff, self.basePosAtt[2] + self.zDiff)
        self.CameraPosAtt = [floor_sensor, self.basePosAtt[5], self.basePosAtt[3], self.basePosAtt[4]]
        #-get floor sensor data

        # +Refresh
        if self.simulCnt % self.refreshRate == 0:
            setModelPosRot(self.posModelIO,
                           self.basePosAtt[0] + self.xDiff, self.basePosAtt[1] + self.yDiff, self.basePosAtt[2] + self.zDiff,
                           0,0,0)
            setModelPosRot(self.rotModelIO,
                           0,0,0,
                           self.basePosAtt[3] + locomotion_pitch, Yaw, self.basePosAtt[5] + locomotion_roll)

            # + drone connecting process
            if self.drone_move_together:
                setModelPosRot(self._posModelIO,
                               self._basePosAtt[0] + self.xDiff, self._basePosAtt[1] + self.yDiff,
                               self._basePosAtt[2] + self.zDiff,
                               0, 0, 0)
                setModelPosRot(self._rotModelIO,
                               0, 0, 0,
                               self._basePosAtt[3] + locomotion_pitch, Yaw, self._basePosAtt[5] + locomotion_roll)
            # - drone connecting process

            if (self.vehicleCamMode != "NONE") and (self.vehicleCamMode == self.VehicleName):
                self.setCamView(Yaw, self.lastPitch - locomotion_pitch, Roll=self.basePosAtt[1] + self.yDiff, Update=(not self.DroneMode))

            # +video
            if self.vidOut is not None:
                Color_image = self.getExtEngineImage()
                cv2.imshow("Color_image", Color_image)
                self.vidOut.write(Color_image)
            # -video

                #+depth file write
                SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()

                Depth_Map = np.zeros((DestHeight, DestWidth), np.float32)
                Depth_Mask = np.zeros((DestHeight, DestWidth, 3), np.uint8)

                t0 = time.monotonic()
                GetDepthMap(Depth_Map.ctypes, Depth_Mask.ctypes, DestWidth, DestHeight, CPUCore, SrcPosX, SrcPosY, SrcWidth, SrcHeight, ObjID)
                t1 = time.monotonic() - t0
                print("Time elapsed: ", t1)

                #Signle File Mode
                SaveRawSingleDepthFile("tmpDataset/DepthMap.bin", Depth_Map)
                #seperateMode
                #SaveRawSeperateDepthFile('tmpDataset/DepthMap%04d.txt' % int(self.simulCnt/self.refreshRate), Depth_Map)

                #-depth file write
                #+ground truth
                with open(self.datasetPath+self.datasetMeta["groundtruth"], 'a') as f:
                    roverBaseID = find_objbyname(self.VehicleName)[0]
                    _, _, posAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
                    posAtt = np.array(posAtt)
                    posAtt -= np.array(self.basePosAtt)
                    f.write(f"{posAtt[0]}\t{posAtt[1]}\t{posAtt[2]}\t{posAtt[3]}\t{posAtt[4]}\t{posAtt[5]}\n")
                # -ground truth
            # -video

            #else:
            #    InitializeRenderFacet(-1, -1)
        # -Refresh

        self.simulCnt += 1
        if self.simulCnt == len(self.motionState) or self.motionState[self.simulCnt] == 3: #state_standby
            self.Simultimer.stop()

            #+video
            if self.vidOut is not None:
                self.vidOut.release()
                self.vidOut = None
            #-video

    def simulCamTiltTimer_slot(self):
        #+locomotion Modeling
        tmpPosDiff = self.leftPos[self.simulCnt] / 1000 + self.basePosAtt[3]
        print(tmpPosDiff)
        #-locomotion Modeling

        if self.simulCnt == 0:
            # +video
            self.vidOut = None
            if self.func3Edit[9].text() != "None":
                fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
                self.vidOut = cv2.VideoWriter(self.func3Edit[9].text() + '.avi', fourcc, 30, (1280, 720))
            # -video



        # +Refresh
        if self.simulCnt % self.refreshRate == 0:
            setModelPosRot(self.rotModelIO,
                           0, 0, 0,
                           tmpPosDiff, self.basePosAtt[4], self.basePosAtt[5])

            if self.vehicleCamMode != "NONE":
                self.setCamView(self.lastYaw, tmpPosDiff)

            #InitializeRenderFacet(-1, -1)

            # +video
            if self.vidOut is not None:
                Color_image = self.getExtEngineImage()
                cv2.imshow("Color_image", Color_image)
                self.vidOut.write(Color_image)
            # -video
        # -Refresh

        self.simulCnt += 1
        if self.simulCnt == len(self.motionState) or self.motionState[self.simulCnt] == 3: #state_standby
            self.Simultimer.stop()

            #+video
            if self.vidOut is not None:
                self.vidOut.release()
                self.vidOut = None
            #-video


    def view_sec1(self):
        self.globalCoordEdit[0].setText("4000")
        self.globalCoordEdit[1].setText("1000")
        self.globalCoordEdit[2].setText("2400.0")
        self.globalCoordEdit[3].setText("0")
        self.globalPosAttSet()
        self.vehicleCamMode = "NONE"

    def view_sec2(self):
        self.globalCoordEdit[0].setText("0")
        self.globalCoordEdit[1].setText("1000")
        self.globalCoordEdit[2].setText("2400.0")
        self.globalCoordEdit[3].setText("0")
        self.globalPosAttSet()
        self.vehicleCamMode = "NONE"

    def view_sec3(self):
        self.globalCoordEdit[0].setText("-4000")
        self.globalCoordEdit[1].setText("1000")
        self.globalCoordEdit[2].setText("2400.0")
        self.globalCoordEdit[3].setText("0")
        self.globalPosAttSet()
        self.vehicleCamMode = "NONE"

    def view_onCam(self):
        self.VehicleName = self.func3Edit[10].text()
        modelLen, modelList = getModelList(1024)
        for mdlIdx, modelName in enumerate(modelList):
            if self.VehicleName+"_CAMERA" == modelName:
                setModelPosRot(0, 0, 0, 0, 0, 0, 0)
                setModelPosRot(1, 0, 0, 0, 0, 0, 0)

                self.camMdlIdx = mdlIdx
                mdlFrame = getLocalFrame(mdlIdx)
                print(mdlFrame)
                tempYaw = -math.atan2(mdlFrame[0][2], mdlFrame[0][0]) * 180. / math.pi + 180
                tempPitch = -math.asin(mdlFrame[2][1]) * 180. / math.pi
                print(tempYaw, tempPitch)
                self.setCamView(tempYaw, tempPitch)
                #InitializeRenderFacet(-1, -1)
                self.vehicleCamMode = self.VehicleName
                break
        #self.globalPosAttSet()

    def clickedWayPnt(self):
        clkX, clkY = GetLastClickPos()
        if clkX != -1 and clkY != -1:
            dist, _, _, _ ,_ , _ = ReturnDistance(clkX, clkY, True)
            px, py, pz = Pixelto3D(clkX, clkY, dist)
            print(clkX,clkY, Pixelto3D(clkX, clkY, dist))
            self.func3Edit[6].setText(str(pz/ 1000))
            self.func3Edit[7].setText("{:.2f}".format(math.atan2(px, pz) / math.pi * 180.).format())

    def greedyNav(self):
        curDir = os.getcwd()
        os.chdir(curDir + "/ThalamusNavigation")

        meterPerPixel = int(self.func4Edit[0].text())
        erodcnt = int(self.func4Edit[1].text())
        gndX = int(self.func4Edit[2].text())
        gndY = int(self.func4Edit[3].text())


        self.greedNavRes, floorMask, ctrPos = getGreedyNav(meterPerPixel, GndPosX=gndX, GndPosY=gndY, dFilename=None, EngNum=1, erodeCnt=erodcnt, compAtt=(-self.lastPitch,0,0))
        if self.greedNavRes is not None:
            self.floorImgList.append(floorMask[0])
            self.floorCtrList.append(ctrPos)

            roverBaseID = find_objbyname(self.VehicleName)[0]
            if self.datasetIndex is None:
                self.datasetIndex = 0
                _, _, self.offsetDSPosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation

            posModelIO, rotModelIO, basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
            datasetAdd = np.array(basePosAtt) - np.array(self.offsetDSPosAtt)
            self.rotposList.append([self.datasetIndex, datasetAdd[0] / 1000, datasetAdd[1] / 1000, datasetAdd[2] / 1000, datasetAdd[3], datasetAdd[4], datasetAdd[5]])
            self.datasetIndex += 1

            for navRes in self.greedNavRes:
                print("Nav res : ", navRes.action, navRes.value)
        else: #there is nothing to move
            self.greedNavRes = [cNavAction(navAction.tilt, 30)]

        os.chdir(curDir)
    #-Function Motion Control

    def naviAction(self):
        if self.greedNavRes is not None:
            if len(self.greedNavRes) == 0:
                print("Complete Local Navigation\n")
            else:
                nav = self.greedNavRes.pop(0)
                print("Nav res : ", nav.action, nav.value)

                if nav.action == navAction.goStraight:
                    self.func3Edit[6].setText(str(nav.value))
                elif nav.action == navAction.rotate:
                    self.func3Edit[7].setText(str(nav.value))

    def navClear(self):
        self.greedNavRes = []

    def mergeMap(self):
        mergeMap, repos = mergeFloor(self.floorImgList, self.floorCtrList, self.rotposList)
        print(repos)
        cv2.imshow("mergeMap", mergeMap)

    def testMvObs(self):
        objID = 7
        obspos = GetObjPos(objID)
        SetObjPos(objID, obspos[0], obspos[1], 600)
        self.movingObsParam = [1600, 3]

    def globalNavi(self):
        meterPerPixel = int(self.func4Edit[0].text())
        tgtX = int(self.func4Edit[4].text())
        tgtY = int(self.func4Edit[5].text())

        #+get Current Positiion
        roverBaseID = find_objbyname(self.VehicleName)[0]
        if self.initBasePosAtt is None:
            _, _, self.initBasePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
        _, _, basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
        curPos = (np.array(basePosAtt) - np.array(self.initBasePosAtt)) / 1000
        #-get Current Positiion

        departPos = getPos2Pix(curPos[0],curPos[2], meterPerPixel)
        departYaw = basePosAtt[4]
        destPos = (tgtX, tgtY)
        filename = "tmpResult/mergeMap.png"

        self.greedNavRes = globalNav(departPos, departYaw, destPos, meterPerPixel, filename)
        for action in self.greedNavRes:
            print(action.action, action.value)


    def getPredictionVertex(self, Color_image):
        # +preprocess
        preprocess1_opt = [
            # (adjustOpt.size_RoiCrop, 100, 200, 500, 500),
            (adjustOpt.size_Resize, 300, 300),  # 0
            (adjustOpt.color_adj, cv2.COLOR_BGR2GRAY)  # 1
        ]
        preprocess1 = getAdjustedImg(Color_image, preprocess1_opt)
        resizedImg = preprocess1[0]
        grayImg = preprocess1[1]
        # -preprocess

        # +context
        cutOffbboxOpt1 = [30, 30, 20, 20]
        featureParams = [featureOpt.corner_GOOD2TRACK, dict(maxCorners=100, qualityLevel=0.1, minDistance=15, blockSize=14)]
        # featureParams = [featureOpt.corner_FAST, 30]
        trackingCriteria = dict(winSize=(15, 15), maxLevel=0, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.03))
        context_opt1 = [(contextOpt.context_diff, 1),
                        (contextOpt.context_optflow, cutOffbboxOpt1, featureParams, trackingCriteria)  # left, right, top, bottom, cut off on boder
                        ]
        imgContext = cimgContext(context_opt=context_opt1)
        # +context

        contextRes = imgContext.inputImg(grayImg)
        contextRes = imgContext.inputImg(grayImg)

        vertexList = []
        if contextRes[1] is not None:
            base_optKey, cur_optKey, idx_opt = contextRes[1]
            optRes = show_optflow(resizedImg.copy(), base_optKey, cur_optKey, idx_opt)
            #cv2.imshow("optical flow", optRes)

            for pos in cur_optKey:
                pos = pos[0]
                pos = [pos[0] * 1280 / 300, pos[1] * 720 / 300]
                dist, _, _, _, _, _ = ReturnDistance(pos[0], pos[1])
                vtx3D = Pixelto3D(pos[0], pos[1], dist)
                # print(pos, dist, vtx3D)
                vertexList.append(vtx3D)
        return vertexList, resizedImg, optRes

    def pedictionGoStraight(self):
        Color_image = self.getExtEngineImage()

        vertexList, resizedImg, optRes = self.getPredictionVertex(Color_image)
        def predictMotion():
            distance = -float(self.func3Edit[6].text())
            SetGlobalPosition(0, 0, distance * 1000)
            InitializeRenderFacet(-1, -1)

        res0, res1 = setPrediction(vertexList, predictMotion)

        showres, angleList, pnt0List, pnt1List = getPredictionVisualization(res0, res1, width=300, height=300, showUnitArrow=False, arrowColor=(0,0,255))

        # showOnce([showres, cv2.add(optRes, showres)], "prediction")
        cv2.imshow("prediction", showres)
        cv2.imshow("prediciton", cv2.add(optRes, showres))



    def pedictionRotate(self):
        Color_image = self.getExtEngineImage()

        vertexList, resizedImg, optRes = self.getPredictionVertex(Color_image)
        def predictMotion():
            angle = -float(self.func3Edit[7].text())
            SetGlobalAttitude(0, angle , 0)
            InitializeRenderFacet(-1, -1)

        res0, res1 = setPrediction(vertexList, predictMotion)

        showres, angleList, pnt0List, pnt1List = getPredictionVisualization(res0, res1, width=300, height=300, showUnitArrow=False, arrowColor=(0,0,255))

        #showOnce([showres, cv2.add(optRes, showres)], "prediction")
        cv2.imshow("prediction", showres)
        cv2.imshow("prediciton", cv2.add(optRes, showres))

    def DetectLocal(self):
        mdlFrame = getLocalFrame(self.camMdlIdx)
        BoundBoxImg, detList = getSimulDetection()
        cv2.imshow("BoundBox", BoundBoxImg)
        for detbbox in detList:
            print(detbbox.bboxCtrX, detbbox.bboxCtrY, detbbox.bboxClass, detbbox.pos3D)
        if len(detList) == 0:
            print("Nothing to detect")
            return

        #+ r
        roverBaseID = find_objbyname(self.VehicleName)[0]

        if self.initBasePosAtt is None :
            _, _, self.initBasePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
        _, _, basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
        basePosAtt = list(np.array(basePosAtt) - np.array(self.initBasePosAtt))


        #+local position save
        offset = [0, -750, 0]  # offset, camera height from rover base
        with open("localMilstones.txt", "w") as f:
            for detbbox in detList:
                f.write("{0} {1} {2} {3}\n".format(detbbox.pos3D[0]+offset[0], detbbox.pos3D[1]+offset[1], detbbox.pos3D[2]+offset[2], detbbox.bboxClass))
        #-local position save

        vertexList = []
        for detbbox in detList:
            vertexList.append(tuple(detbbox.pos3D))
        def predictMotion():
            SetGlobalAttitude(-self.lastPitch, basePosAtt[4], basePosAtt[5])
            InitializeRenderFacet(-1, -1)

        res0, res1 = setPrediction(vertexList, predictMotion)
        #print(res0)
        #print(res1)
        tmpPos = []
        for idx, detbbox in enumerate(detList):
            res = res1[idx]
            pos = [basePosAtt[0]+res[0], basePosAtt[1]+res[1], basePosAtt[2]+res[2]]
            tmpPos.append(pos)

            closeKey = searchClosePnt(self.milestoneList, pos, 500)
            if -1 == closeKey:
                self.milestoneList.append(detMilestne(pos, detbbox.bboxClass))
            else: #Update
                self.milestoneList[closeKey].pos3D[0] = (self.milestoneList[closeKey].pos3D[0] + pos[0]) / 2
                self.milestoneList[closeKey].pos3D[1] = (self.milestoneList[closeKey].pos3D[1] + pos[1]) / 2
                self.milestoneList[closeKey].pos3D[2] = (self.milestoneList[closeKey].pos3D[2] + pos[2]) / 2

        #+write file with offset
        with open("milstones.txt", "w") as f:
            for milestone in self.milestoneList:
                print(milestone.pos3D, milestone.classID)
                f.write("{0} {1} {2} {3}\n".format(milestone.pos3D[0] + offset[0], milestone.pos3D[1] + offset[1], milestone.pos3D[2] + offset[2], milestone.classID))
        #- write file with offset

        #+Display Result
        npTmpPos = np.array(tmpPos)

        xAvgPix = int(np.average(npTmpPos[:, 0]) / 1000 * 50)
        yAvgPix = int(np.average(npTmpPos[:, 1]) / 1000 * 50)

        msMap = np.zeros((1000, 1000, 3), np.uint8)
        for milestone in self.milestoneList:
            x = int(milestone.pos3D[0] / 1000 * 50)
            y = int(-milestone.pos3D[2] / 1000 * 50)

            color = (0,0,0)
            if milestone.classID == 0:
                color = (255,255,255)
            if milestone.classID == 1:
                color = (255, 0, 0)
            if milestone.classID == 2:
                color = (0, 255, 0)
            if milestone.classID == 3:
                color = (255, 255, 0)
            if milestone.classID == 4:
                color = (255, 0, 255)
            cv2.drawMarker(msMap, (500 + x - xAvgPix, 500 + y - yAvgPix), color=color, markerType=cv2.MARKER_CROSS, thickness=1, markerSize=15)

        cv2.imshow("milestone Map", msMap)
        #-Display Result
    def locomotionImidiatly(self, distance=0, angle=0):
        self.VehicleName = self.func3Edit[10].text()
        roverBaseID = find_objbyname(self.VehicleName)[0]
        self.posModelIO, self.rotModelIO, self.basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation

        Yaw = self.basePosAtt[4] + angle
        self.xDiff = distance * self.timeSlice * math.sin(Yaw * math.pi / 180.)
        self.zDiff = distance * self.timeSlice * math.cos(Yaw * math.pi / 180.)

        setModelPosRot(self.posModelIO,
                       self.basePosAtt[0] + self.xDiff, self.basePosAtt[1], self.basePosAtt[2] + self.zDiff,
                       0, 0, 0)
        setModelPosRot(self.rotModelIO,
                       0, 0, 0,
                       self.basePosAtt[3], Yaw, self.basePosAtt[5])

        if self.vehicleCamMode != "NONE":
            self.setCamView(Yaw, self.lastPitch)

        # + drone connecting process
        self.vehicle_obj = find_objbyname(self.VehicleName)
        self.drone_move_together = False
        if -1 != self.VehicleName.lower().find("rover"):
            self._vehicle_obj = find_objbyname("Drone1")
            droneBaseID = self._vehicle_obj[0]
            self._posModelIO, self._rotModelIO, self._basePosAtt = self.getSrcPosAtt(droneBaseID, -4,
                                                                                     -3)  # get Src Position / Rotation
            collision_list = GetCollision(self._vehicle_obj[0], self._vehicle_obj)
            for objid in self.vehicle_obj:
                if objid in collision_list:
                    self.drone_move_together = True
        # - drone connecting process
        # + drone connecting process
        if self.drone_move_together:
            setModelPosRot(self._posModelIO,
                           self._basePosAtt[0] + self.xDiff, self._basePosAtt[1],
                           self._basePosAtt[2] + self.zDiff,
                           0, 0, 0)
            setModelPosRot(self._rotModelIO,
                           0, 0, 0,
                           self._basePosAtt[3], Yaw,
                           self._basePosAtt[5])
        # - drone connecting process

        InitializeRenderFacet(-1, -1)
    def goImidiatly(self):
        distance = float(self.func3Edit[6].text()) * 1000 #mm 2 Meter
        self.locomotionImidiatly(distance=distance)
    def rotImidiatly(self):
        angle = float(self.func3Edit[7].text())
        self.locomotionImidiatly(angle=angle)
    def tiltImidiatly(self):
        camBaseID = 3
        self.posModelIO, self.rotModelIO, self.basePosAtt = self.getSrcPosAtt(camBaseID, -2, -1)  # get Src Position / Rotation
        angle = float(self.func3Edit[8].text())

        setModelPosRot(self.rotModelIO, 0, 0, 0, angle, self.basePosAtt[4], self.basePosAtt[5])

        if self.vehicleCamMode != "NONE":
            self.setCamView(self.lastYaw, angle)

        InitializeRenderFacet(-1, -1)
        pass

if __name__ == '__main__':
    print(cv2.__version__)
    print(os.getcwd())

    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())


