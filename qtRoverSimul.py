
import sys
import cv2
import os
import numpy as np
import time
import math

from ThalamusEngine.Interface import *
from matplotlib import pyplot as plt
from MCLib.mcInterface import *

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


class Window(QWidget):
    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        mainGrid = QGridLayout()

        label = [["ScriptFile:", "EngineName:"]]
        editDefault = [["ScriptRover.txt", "Thalamus QT Example"]]
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
        buttonText = ["DepthMap", "ColorMap", "NoShade", "LightEff", "Bounding Box", "ext EngColor", "DataSet Adding"]
        buttonFunc = [self.funcDepthMap, self.funcColorMap, self.funcNoShade, self.funcLightEffect, self.funcBBox,
                      self.funcExtEngineViewMap, self.funcDatasetAdding]
        subgrid, self.func1Edit = self.createGroupBox("Scene Generation", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 4, 0)

        label = [["DepthMap:", "Width", "Height", "MeshUp Inv", "FreeModelNum", "Thread"],["ColorImg:"]]
        editDefault = [["depthmap.txt","300", "300", "9", "5", "12"],["Dataset03/Color03.png"]]
        buttonText = ["MeshUp", "Texture Overay", "Texure Int", "TextureView"]
        buttonFunc = [self.func2MeshUp, self.func2TexOveray, self.func2TexInt, self.func2TexView]
        subgrid, self.func2Edit = self.createGroupBox("Mesh up, Texture Overay", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 5, 0)


        label = [["axis", "acc", "cruiseSpd", "dcc", "distance", "Simul Len"], ["Straight", "Rotate", "Cam Tilt"]]
        editDefault = [["0", "1.0", "1.0", "1.0", "3.0", "5.0"], ["1.0",  "45.0", "0"]]
        buttonText = ["getProfile", "go Straigt", "Rotate", "cam Tilt", "View Sec1", "View Sec2", "View Sec3", "View Cam", "Clicked WayPnt"]
        buttonFunc = [self.motion_getProfile, self.motion_goFoward, self.motion_Rotate, self.motion_CamTilt,
                      self.view_sec1, self.view_sec2, self.view_sec3, self.view_onCam, self.clickedWayPnt]
        subgrid, self.func3Edit = self.createGroupBox("Motion Control", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 6, 0)

        label = [["1meter/pix", "moveable area", "GndPosX", "GndPosY"]]
        editDefault = [["50"," 25", "29", "31"]]
        buttonText = ["greedy Nav", "Navi Action", "Nav Clear", "MergeMap"]
        buttonFunc = [self.greedyNav, self.naviAction, self.navClear, self.mergeMap ]
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
        self.wheelCircum = math.pi * (0.22 + 0.03)
        # -Motion Constant

        #+Flag For Application
        self.vehicleCamMode = False
        #-Flag For Application

        #+Dataset setup
        self.datasetIndex = None
        #-Dataset setup

        #+Navagation value
        self.greedNavRes = None
        self.floorImgList = []
        self.floorCtrList = []
        self.rotposList = []
        #-Navagation value

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
        InitSimulation(AsmFileName, SimWindowText)
        pass

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
        self.vehicleCamMode = False

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

        SaveRawDepthFile('depthmap.txt', Depth_Map)

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

    def funcExtEngineViewMap(self):
        Color_width = 1280
        Color_Height = 720
        Color_image = np.zeros((Color_Height, Color_width, 3), np.uint8)
        InitializeRenderFacet(-1, -1)  # refresh
        GetColorImage(Color_image.ctypes, Color_width, Color_Height)
        Color_image = cv2.resize(Color_image, (300, 300), cv2.INTER_LANCZOS4)
        cv2.imshow("External Engine Color Image", Color_image)
        cv2.imwrite("extColor.png", Color_image)

    def funcDatasetAdding(self):
        roverBaseID = 1
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
        SaveRawDepthFile(fname, Depth_Map)
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
    def func2MeshUp(self):

        depWidth = int(self.func2Edit[1].text())
        depHeight = int(self.func2Edit[2].text())
        depInv = int(self.func2Edit[3].text())
        MeshUpType = int(self.func2Edit[4].text())

        Depth_Map = np.zeros((depHeight, depWidth), np.float32)
        Depth_Mask = np.zeros((depHeight, depWidth, 3), np.uint8)

        if 0 != LoadBinDepthMapPnt(self.func2Edit[0].text(), depWidth, depHeight, 600, 6000, Depth_Map.ctypes, Depth_Mask.ctypes):
            Depth_Map = cv2.normalize(Depth_Map, None, alpha=0, beta=1.0, norm_type=cv2.NORM_MINMAX)
            cv2.imshow("Depth Map", Depth_Map)
            ret = ObjMeshUp(depWidth, depHeight, MeshUpType, depInv)
            print(ret)
            InitializeRenderFacet(-1, -1)
        else:
            print("Loading Error")
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
        timeSlice = 10
        print(MoveDS(axisIdx, acc, cruiseSpd, dcc, distance))  # acc, cruise_spd, dcc, distance
        ret, Time, Spd, Pos, State = getProfile(axisIdx, simulTime, timeSlice)


        plt.plot(Time, Spd, label="spd")
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
        self.timeSlice = 10
        ret, Time, Spd, Pos, State = getProfile(axisIdx, simulTime, self.timeSlice)
        self.leftPos = leftSign * Pos.copy()
        self.rightPos = rightSign * Pos.copy()
        self.leftSpd = leftSign * Spd.copy()
        self.rightSpd = rightSign * Spd.copy()
        self.motionState = State
        # -get profile

    def startSimulTimer(self, timerFunc):
        # +start Simul Timer
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
        if 0 < distance:
            self.motionInit(1, 1, distance)
        else:
            self.motionInit(-1, -1, -distance)
        self.startSimulTimer(self.simulLocoTimer_slot)
    def motion_Rotate(self):
        angle = float(self.func3Edit[7].text())
        distance = angle / 180. * self.wheelCircum #Unit:M
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

    def setCamView(self, Yaw, Pitch):
        # +Initiaizize Pos/Att
        setModelPosRot(0, 0, 0, 0, 0, 0, 0)
        setModelPosRot(1, 0, 0, 0, 0, 0, 0)
        # -Initiaizize Pos/Att

        mdlFrame = getLocalFrame(self.camMdlIdx)
        CamSz = 30 # Cam depth(size Z) is 30, somwhat margin 15->20 casue of pitch movement
        camOffsetX = CamSz * math.sin(Yaw * math.pi / 180.)
        camOffsetZ = CamSz * math.cos(Yaw * math.pi / 180.)

        setModelPosRot(0, 0,0,0, Pitch, 0, 0)
        setModelPosRot(1, -(mdlFrame[3][0] + camOffsetX), -mdlFrame[3][1], -(mdlFrame[3][2] + camOffsetZ), 0, -Yaw, 0)
        self.lastPitch = Pitch #for locomotion Timer's Initial Value
        self.lastYaw = Yaw #for locomotion Timer's Initial Value
    def simulLocoTimer_slot(self):
        #Initialize Movement
        if self.simulCnt == 0:
            roverBaseID = 1
            self.posModelIO, self.rotModelIO, self.basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3) #get Src Position / Rotation
            self.zDiff = 0
            self.xDiff = 0
        #+locomotion Modeling

        #+Calculation Yaw
        tmpDiff = (self.leftPos[self.simulCnt] - self.rightPos[self.simulCnt]) / 2

        rotateLoop = 2 * self.wheelCircum * 1000
        tmpSign = np.sign(tmpDiff)
        tmpDiff -= int(tmpDiff / rotateLoop) * rotateLoop * tmpSign
        Yaw = tmpDiff / rotateLoop * 360. + self.basePosAtt[4]
        # -Calculation Yaw

        tmpPosDiff = (self.leftSpd[self.simulCnt] + self.rightSpd[self.simulCnt]) / 2
        self.xDiff += tmpPosDiff * self.timeSlice * math.sin(Yaw * math.pi / 180.)
        self.zDiff += tmpPosDiff * self.timeSlice * math.cos(Yaw * math.pi / 180.)
        print(Yaw, tmpPosDiff)
        #-locomotion Modeling

        # +Refresh
        if self.simulCnt % self.refreshRate == 0:
            setModelPosRot(self.posModelIO,
                           self.basePosAtt[0] + self.xDiff, self.basePosAtt[1], self.basePosAtt[2] + self.zDiff,
                           0,0,0)
            setModelPosRot(self.rotModelIO,
                           0,0,0,
                           self.basePosAtt[3], Yaw, self.basePosAtt[5])

            if self.vehicleCamMode:
                self.setCamView(Yaw, self.lastPitch)

            InitializeRenderFacet(-1, -1)
        # -Refresh

        self.simulCnt += 1
        if self.simulCnt == len(self.motionState) or self.motionState[self.simulCnt] == 3: #state_standby
            self.Simultimer.stop()

    def simulCamTiltTimer_slot(self):
        #+locomotion Modeling
        tmpPosDiff = self.leftPos[self.simulCnt] / 1000 + self.basePosAtt[3]
        print(tmpPosDiff)
        #-locomotion Modeling

        # +Refresh
        if self.simulCnt % self.refreshRate == 0:
            setModelPosRot(self.rotModelIO,
                           0, 0, 0,
                           tmpPosDiff, self.basePosAtt[4], self.basePosAtt[5])

            if self.vehicleCamMode:
                self.setCamView(self.lastYaw, tmpPosDiff)

            InitializeRenderFacet(-1, -1)
        # -Refresh

        self.simulCnt += 1
        if self.simulCnt == len(self.motionState) or self.motionState[self.simulCnt] == 3: #state_standby
            self.Simultimer.stop()

    def view_sec1(self):
        self.globalCoordEdit[0].setText("4000")
        self.globalCoordEdit[1].setText("1000")
        self.globalCoordEdit[2].setText("2400.0")
        self.globalCoordEdit[3].setText("0")
        self.globalPosAttSet()
        self.vehicleCamMode = False

    def view_sec2(self):
        self.globalCoordEdit[0].setText("0")
        self.globalCoordEdit[1].setText("1000")
        self.globalCoordEdit[2].setText("2400.0")
        self.globalCoordEdit[3].setText("0")
        self.globalPosAttSet()
        self.vehicleCamMode = False

    def view_sec3(self):
        self.globalCoordEdit[0].setText("-4000")
        self.globalCoordEdit[1].setText("1000")
        self.globalCoordEdit[2].setText("2400.0")
        self.globalCoordEdit[3].setText("0")
        self.globalPosAttSet()
        self.vehicleCamMode = False

    def view_onCam(self):
        modelLen, modelList = getModelList(1024)
        for mdlIdx, modelName in enumerate(modelList):
            if "Rover1_CAMERA" == modelName:

                setModelPosRot(0, 0, 0, 0, 0, 0, 0)
                setModelPosRot(1, 0, 0, 0, 0, 0, 0)

                self.camMdlIdx = mdlIdx
                mdlFrame = getLocalFrame(mdlIdx)
                print(mdlFrame)
                tempYaw = -math.atan2(mdlFrame[0][2], mdlFrame[0][0]) * 180. / math.pi + 180
                tempPitch = -math.asin(mdlFrame[2][1]) * 180. / math.pi
                print(tempYaw, tempPitch)
                self.setCamView(tempYaw, tempPitch)
                InitializeRenderFacet(-1, -1)
                self.vehicleCamMode = True
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
            self.floorImgList.append(floorMask)
            self.floorCtrList.append(ctrPos)

            roverBaseID = 1
            if self.datasetIndex is None:
                self.datasetIndex = 0
                _, _, self.offsetDSPosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation

            posModelIO, rotModelIO, basePosAtt = self.getSrcPosAtt(roverBaseID, -4, -3)  # get Src Position / Rotation
            datasetAdd = np.array(basePosAtt) - np.array(self.offsetDSPosAtt)
            self.rotposList.append([self.datasetIndex, datasetAdd[0] / 1000, datasetAdd[1] / 1000, datasetAdd[2] / 1000, datasetAdd[3], datasetAdd[4], datasetAdd[5]])
            self.datasetIndex += 1
            #

            for navRes in self.greedNavRes:
                print("Nav res : ", navRes.action, navRes.value)

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

        mergeMap = mergeFloor(self.floorImgList, self.floorCtrList, self.rotposList)
        cv2.imshow("mergeMap", mergeMap)
        pass

if __name__ == '__main__':
    print(cv2.__version__)
    print(os.getcwd())

    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())


