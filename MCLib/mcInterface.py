

import numpy as np
import ctypes


def InitMCSimul():
    TestDLL = ctypes.CDLL('./MCLib/MotionControl.so')

    global InitContollerFunc
    InitContollerFunc = TestDLL['InitContoller']
    InitContollerFunc.argtypes = (ctypes.c_float,)
    InitContollerFunc.restype = ctypes.c_int

    global MoveDSFunc
    MoveDSFunc = TestDLL['MoveDS']
    MoveDSFunc.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_float, ctypes.c_float, ctypes.c_float,)
    MoveDSFunc.restype = ctypes.c_int

    global GetProfileFunc
    GetProfileFunc = TestDLL['GetProfile']
    GetProfileFunc.argtypes = (ctypes.c_int, ctypes.c_float, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,)
    GetProfileFunc.restype = ctypes.c_int

    global getCurrentStateFunc
    getCurrentStateFunc = TestDLL['GetCurrentState']
    getCurrentStateFunc.argtypes = (ctypes.c_int, ctypes.c_void_p,)
    getCurrentStateFunc.restype = ctypes.c_int

def InitContoller(p1):
    return InitContollerFunc(p1)

def MoveDS(p1, p2, p3, p4, p5):
    return MoveDSFunc(p1,p2,p3,p4,p5)

#int GetProfile(int axisIdx, float SimulTime, int *Time, float *Spd, float *Pos, unsigned char* State)
def getProfile(axisIdx, SimulTime):
    timeSlice = 1 #matching with #define TIME_SLICE 1
    bufferLen = int(float(SimulTime) * 1000 / float(timeSlice))
    Time = np.zeros(bufferLen, np.int32)
    Spd = np.zeros(bufferLen, np.float32)
    Pos = np.zeros(bufferLen, np.float32)
    State = np.zeros(bufferLen, np.uint8)
    ret = GetProfileFunc(axisIdx, SimulTime, Time.ctypes, Spd.ctypes, Pos.ctypes, State.ctypes)
    return ret, Time, Spd, Pos, State

def getCurrentState(axisIdx):
    buff = np.zeros(2, np.float32)
    ret = getCurrentStateFunc(axisIdx, buff.ctypes)
    return ret, buff[0], buff[1] #state, spd, pos