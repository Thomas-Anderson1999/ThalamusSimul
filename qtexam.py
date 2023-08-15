
import sys
import cv2
import os
import numpy as np
import time

from ThalamusEngine.Interface import *


#+UI
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
#-UI

import sys
from PyQt5.QtWidgets import QApplication, QWidget


class Window(QWidget):

    def __init__(self, parent=None):
        super(Window, self).__init__(parent)

        mainGrid = QGridLayout()

        label = [["ScriptFile:", "EngineName:"]]
        editDefault = [["Script.txt", "Thalamus QT Example"]]
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

        label = [["ObjID", "ModelID", "Offset"], ["PosX", "PosY", "PosZ", "AttX", "AttY", "AttZ"]]
        editDefault = [["-1", "0", "-1"], ["0", "0", "0", "0", "0", "0"]]
        buttonText = ["MoelGetParam", "Param Set"]
        buttonFunc = [self.mdlGetParam, self.mdlSetParam]
        subgrid, self.mdlControlEdit = self.createGroupBox("Modeiling Control", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 3, 0)

        label = [["SrcPosX", "SrcPosY", "SrcWidth", "SrcHeight", "DestWidth", "DestHeight"], ["ObjID", "CPU Core"]]
        editDefault = [["0", "0", "1280", "720", "300", "300"], ["-1", "12"]]
        buttonText = ["DepthMap", "ColorMap", "NoShade", "LightEff", "Bounding Box"]
        buttonFunc = [self.funcDepthMap, self.funcColorMap, self.funcNoShade, self.funcLightEffect, self.funcBBox]
        subgrid, self.func1Edit = self.createGroupBox("Scene Generation", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 4, 0)

        label = [["DepthMap:", "Width", "Height", "MeshUp Inv", "FreeModelNum", "Thread"],["ColorImg:"]]
        editDefault = [["Dataset03/DepthBin03.txt","300", "300", "9", "4", "12"],["Dataset03/Color03.png"]]
        buttonText = ["MeshUp", "Texture Overay", "Texure Int", "TextureView"]
        buttonFunc = [self.func2MeshUp, self.func2TexOveray, self.func2TexInt, self.func2TexView]
        subgrid, self.func2Edit = self.createGroupBox("Mesh up, Texture Overay", label, editDefault, buttonText, buttonFunc)
        mainGrid.addWidget(subgrid, 5, 0)

        self.setLayout(mainGrid)
        self.setWindowTitle("Thalamus Engine UI")

        self.resize(600, -1)

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
        SetGlobalPosition(x,y,z)
        InitializeRenderFacet(-1, -1)
    def globalAttSet(self):
        x = float(self.globalCoordEdit[3].text())
        y = float(self.globalCoordEdit[4].text())
        z = float(self.globalCoordEdit[5].text())
        SetGlobalAttitude(x, y, z)
        InitializeRenderFacet(-1, -1)
    def globalPosAttSet(self):
        self.globalPosSet()
        self.globalAttSet()
    def globalBirdview(self):
        self.globalCoordEdit[1].setText("8000")
        self.globalCoordEdit[2].setText("-3000")
        self.globalCoordEdit[3].setText("90")
        self.globalPosAttSet()

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

    def funcNoShade(self):
        SrcPosX, SrcPosY, SrcWidth, SrcHeight, DestWidth, DestHeight, ObjID, CPUCore = self.getFunc1Param()

        Shade_Mask = np.zeros((DestHeight, DestWidth, 3), np.uint8)
        Shade_Img = np.zeros((DestHeight, DestWidth, 3), np.uint8)

        GetColorImageNoShade(Shade_Img.ctypes, Shade_Mask.ctypes, DestWidth, DestHeight, CPUCore, SrcPosX, SrcPosY, SrcWidth, SrcHeight)
        cv2.imshow("Shade_Img", Shade_Img)
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
if __name__ == '__main__':
    print(cv2.__version__)
    print(os.getcwd())

    app = QApplication(sys.argv)
    clock = Window()
    clock.show()
    sys.exit(app.exec_())

