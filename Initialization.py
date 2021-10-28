import numpy as np
import math
import time
import random
import scipy.spatial.distance


def initialization(path1, path2, param, model):
    # 0bx 1by 2xp 3yp 4ht 5wd 6sc 7xi 8yi 9xw 10yw
    DSE = 2
    XYZ = [[[], []], [[], []]]
    k = 0
    detections_two = path2
    detections_one = path1
    if len(detections_one) and len(detections_two):

        XYZ[0][0] = [i for i, j in zip(detections_one[1], detections_one[5]) if j > param['Prun_Thre']]  # xi of first frame
        XYZ[0][1] = [i for i, j in zip(detections_one[2], detections_one[5]) if j > param['Prun_Thre']]  # yi of first frame
        XYZ[1][0] = [i for i, j in zip(detections_two[1], detections_two[5]) if
                     j > param['Prun_Thre']]  # xi of second frame
        XYZ[1][1] = [i for i, j in zip(detections_two[2], detections_two[5]) if
                 j > param['Prun_Thre']]  # yi of second frame
        # print(XYZ[0][0])
        T = model['T']
        Vmax = param['Vmax']
        aaa = model['H']
        DMV = len(aaa)
        X0 = np.zeros((len(XYZ[0][0]), len(model['F'])))
        i = 0

        # j = 0
        for item in list(range(DSE)):

            if len(XYZ[0]) != 0:
                if i == 0:
                    j = 0
                    for it in list(range(DMV)):
                        # print()
                        # print(XYZ[i][j])
                        X0[:, j * DSE + i] = XYZ[i][j]
                        j += 1
                else:
                    temp = scipy.spatial.distance.cdist(np.transpose(np.array(XYZ[i])), np.transpose(np.array(XYZ[0])))
                    mini = np.argmin(temp, 0)
                    # np.mini
                    XYZ2M = np.transpose(np.array(XYZ[i]))[mini]
                    # XYZ2M = [np.transpose(np.array(XYZ[i]))[item_o] for item_o in mini]
                    k = 0
                    for jtem in list(range(DMV)):
                        XYZ2N = X0[:, k * DSE]

                        X0[:, k * DSE + i] = (XYZ2M[:, k] - XYZ2N) / T
                        # print(abs(X0[:, k * DSE + i])<Vmax)
                        X0[:, k * DSE + i] = (abs(X0[:, k * DSE + i]) < Vmax) * X0[:, k * DSE + i]
                        k += 1

            else:
                if i == 1:
                    X0 = []
                    print('Nothing detected in the first frame')
            i += 1

    else:
        X0 = []
    return X0
