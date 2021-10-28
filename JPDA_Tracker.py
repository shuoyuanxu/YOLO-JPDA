# This is the main function which takes the detections, model and
# parameters and generates trajectories.
# ----------------------------------------------------------------------------------------------
# Output:
# Xe: Estimated mean of targets' states representing the trajectories
# - position and velocity at each frame (x_t Vx_t y_t Vy_t)
# Pe: Estimated covariance of the targets' states
# Ff: Frame numbers "t" for which each target's state is estimated.
# Term_Con: Number of consecutive frames which a target is miss-detected before its termination
# M: used to generate video clip
# ----------------------------------------------------------------------------------------------
# Input:
# Detection_address: The path to load the detections
# param: The method's parameters
# model: Tracking models
# Image_address: The path to load the image frames for visualization
# ----------------------------------------------------------------------------------------------
import numpy as np
import time
import scipy.sparse.csgraph
import random
from JPDA_Probabilty_Calculator import JPDA_Probabilty_Calculator
from Tree_Constructor import Tree_Constructor


def ApproxMultiscanJPDAProbabilities(M, Obj_info, mbest):
    U = len(Obj_info)
    Final_probabilty = []
    for i in range(19):
        Final_probabilty.append([])

    n_components, labels = scipy.sparse.csgraph.connected_components(M, False, 'strong')
    C2 = labels[:U]

    for i in np.unique(C2):
        ix = [item for item in range(len(C2)) if C2[item] == i]
        tempInput = [obj_info[item] for item in ix]
        temp = JPDA_Probabilty_Calculator(tempInput)
        for j in ix:
            Final_probabilty[j] = temp[ix.index(j)]


def ismember(a, b):
    if a in b:
        return False
    else:
        return True


def cellfundevidedsum(b, stepNo):
    if b == []:
        return b
    else:
        if type(b) == float:
            c = np.array(1).reshape(-1, 1)
            # return c
        else:

            c = b / float(sum(b))
            # c = np.array([item / sum(b) for item in b]).reshape(-1)
        return c


def JPDA(path, model, param, X0, P0, Term_Con, stepNo, Terminated_objects_index, Image_address=None):
    # if stepNo%50==0:
    print('stepNo now is:', stepNo)

    # check the number of input arguments e.g .3 <= n_input <= 4
    detection_new = path

    Frame = 2  # len(detections[2]) Total Number of frames

    # Detection parameter
    Prun_Thre = param['Prun_Thre']  # Parameter for pruning detections with the confidence score less than this value

    # Track termination parameter
    Term_Frame = param['Term_Frame']  # The parameter for termination condition

    # Kalman Models
    F = model['F']  # The transition matrix for the dynamic model
    Q = model['Q']  # The process covariance matrix for the dynamic model
    H = model['H']  # Measurement matrix
    R = model['R']  # Measurement covariance matrix
    P1 = model['P0']

    # JPDA Parameters
    PD = param['PD']  # Detection Probability
    Beta = param['Beta']  # False detection (clutter) likelihood
    Gate = param['Gate']  # Gate size for gating
    S_limit = param['S_limit']  # parameter to stop the gate size growing too much
    # mbest = param['N_H']  # Threshold for considering m-best number of Hypotheses for JPDA

    DMV = len(H)
    DSV = len(H[0])
    if not(len(X0)):
        N_Target = 0
    else:
        N_Target = len(np.array(X0)[0])
    # print(N_Target)
    # ---------------------------------- Parameters from Last Frame -------------------------------------
    # Initial State Vector
    Xe = []
    # Pe = []
    BxT = []
    Ff = []

    ij = 0
    for index in list(range(N_Target)):
        Ff.append([1, 1])
        ij += 1
    Pe = P0
    Xe = X0
    # print(np.array(X0).shape,'\n')
    Xe = list(np.transpose(Xe))

    # ---------------- loading Image and detections for frame number > 1 -------------------------
    f = 1  # previous frame
    if f >= 1:  # previous frame
        ordinaryFrame = []
        if not(len(detection_new)):
            ordinaryFrame = []
        else:
            aaa = [detection_new[1][item] for item in range(len(detection_new[5])) if
               detection_new[5][item] > Prun_Thre]
            bbb = [detection_new[2][item] for item in range(len(detection_new[5])) if
               detection_new[5][item] > Prun_Thre]
            ordinaryFrame.append(aaa)
            ordinaryFrame.append(bbb)
            ordinaryFrame = np.transpose(ordinaryFrame)

    # N_Target = len(Xe[1])
    MXe = np.zeros((DSV, N_Target))
    PXe = np.zeros((DSV, DSV, N_Target))
    S = np.zeros((DMV, DMV, N_Target))
    K = np.zeros((DSV, DMV, N_Target))
    Target_Obs_indx = [[] for item in range(N_Target)]
    Target_probabilty = [[] for item in range(N_Target)]
    Curr_Obs = []
    Xe = np.array(Xe)
    Pe = np.array(Pe)
    # print(Pe, '\n')
    # if stepNo >= 88:
    #     print('1234567890:',N_Target)
    # if stepNo == 99:
    #     print('1111:', Pe)
    for no in list(range(N_Target)):
        # if stepNo == 99:
        #     print(N_Target, Pe[no])
        if not (no in Terminated_objects_index):
            # Kalman Preditction & Hypothesis Construction
            Target_Obs_indx[no], Target_probabilty[no], MXe[:, no], PXe[:, :, no], S[:, :, no], K[:, :, no] = \
                Tree_Constructor(Xe[no].reshape(4), Pe[no], F, Q, H, R, ordinaryFrame, S_limit, Gate, PD, Beta, DMV,
                                 stepNo, no)
            #     Mes_Tar[Target_Obs_indx[no], no] = true
            # print(N_Target)
            # print(Target_Obs_indx[no])

            # if stepNo == 157:
            #     print(no, Target_probabilty[no])

            # if stepNo == 157:
            #     print(Target_Obs_indx[58])
    Temp_probability = [[] for count in range(N_Target)]
    exist_ind = [ismember(item_temp, Terminated_objects_index) for item_temp in range(N_Target)]
    # Temp_probability = [item_temptemp for item_temptemp, item_temptemp_index in zip(Target_probabilty, exist_ind) if
    #                     item_temptemp_index]
    for item_temp in range(N_Target):
        if exist_ind[item_temp]:
            Temp_probability[item_temp] = Target_probabilty[item_temp]
    # Temp_probability = [Target_probabilty[exist_ind.index()] for item_temptemp_index in exist_ind if item_temptemp_index]
    ##############################################
    # if stepNo == 157:
    #    print(Temp_probability[57],)
    ################################################

    Final_probabilty = np.array([cellfundevidedsum(item_prob, stepNo) for item_prob in Temp_probability])
    # **************************** Update step *****************************
    for no in range(N_Target):

        if not (no in Terminated_objects_index):
            # k = f - Ff[no][0, 0] + 1
            NN = len(Target_Obs_indx[no])

            # if stepNo == 157:
            #     print(no,Final_probabilty[no])

            P_temp = np.transpose(Final_probabilty[no])
            xsum = np.zeros((4, 1))

            if not Target_Obs_indx[no]:
                # if len(P_temp[0]) == 1:
                Xe[no] = MXe[:, no]
                dP = 0
                Pe[no] = PXe[:, :, no]
            else:
                Yij = ordinaryFrame[Target_Obs_indx[no]] - np.tile(np.transpose(np.dot(H, MXe[:, no])),
                                                                   [np.size(Target_Obs_indx[no]), 1])

                # if stepNo == 157:
                #     print(Target_Obs_indx[58])

                # Ye = P_temp[0][NN] * Yij
                # if len(P_temp[0]) == 1:
                #    Ye = Yij
                #     # Ye = Ye.reshape((1, -1))
                #
                #     if stepNo == 157:
                #         print(Ye)
                #
                #     Xe[no] = (MXe[:, no]).T + (np.dot(K[:, :, no], Ye.T)).T
                #     dP = np.dot(
                #         np.dot(K[:, :, no], (np.dot(np.tile(P_temp[0], [DMV, 1]) * Yij.T, Yij) - np.dot(Ye.T, Ye))),
                #         K[:, :, no].T)
                #     # print('215:', Ye, '\n')
                # if stepNo == 157:
                #     print(no,P_temp[0],Yij)
                # else:
                Ye = np.dot(P_temp[0][1:], Yij)
                Ye = Ye.reshape((1, -1))

                # if stepNo == 157:
                #     print(Ye)

                # print('213:', Ye, '\n')
                # print(Xe.shape)
                Xe[no] = (MXe[:, no]).T + (np.dot(K[:, :, no], Ye.T)).T
                dP = np.dot(
                    np.dot(K[:, :, no], (np.dot(np.tile(P_temp[0][1:], [DMV, 1]) * Yij.T, Yij) - np.dot(Ye.T, Ye))),
                    K[:, :, no].T)

            Pst = PXe[:, :, no] - np.dot(np.dot(K[:, :, no], S[:, :, no]), K[:, :, no].T)
            # print(PXe[:, :, no],P_temp,no,'\n')
            if len(P_temp.reshape(-1, 1)) != 1:
                # if stepNo == 157:
                #     print(P_temp)
                Po = P_temp[0][0] * PXe[:, :, no] + (1 - P_temp[0][0]) * Pst
            else:
                Po = P_temp * PXe[:, :, no] + (1 - P_temp) * Pst
            Pe[no] = Po + dP
            # if stepNo == 98:
            #     print(Pe[no])
            # print(Pe, '\n')


            # indM = np.argmax(P_temp)
            if np.argmax(P_temp.reshape(-1, 1)) == 0:
                Term_Con[no] += 1
            else:
                Term_Con[no] = 0
                # print()
            # print(np.array(Curr_Obs).shape, np.array(Target_Obs_indx[no]).shape,'\n')
            if Target_Obs_indx[no]:
                if Curr_Obs == []:

                    Curr_Obs = Target_Obs_indx[no][0]
                else:
                    Curr_Obs = np.vstack(
                        (np.array(Curr_Obs).reshape(-1, 1), np.array(Target_Obs_indx[no]).reshape(-1, 1)))

            if (Term_Con[no] <= Term_Frame) or (Term_Con[no] == 0):
                pass
                # Ff[no][1, 2] = f
            else:

                if Terminated_objects_index:

                    Terminated_objects_index.append(no)
                else:
                    Terminated_objects_index = [no]
    # print(Terminated_objects_index, Curr_Obs)
    # if stepNo == 155:
    #     print(Terminated_objects_index)
    # if stepNo == 156:
    #     print(Terminated_objects_index)
    #    # print(len(Term_Con))
    # if stepNo >= 156:
    #     print(Terminated_objects_index, '1234:', stepNo)
    All_Obs = len(ordinaryFrame)
    if type(Curr_Obs) != np.ndarray:
        Curr_Obs_1 = [Curr_Obs]
    else:
        Curr_Obs_1 = Curr_Obs
    New_Targets = [item for item in range(All_Obs) if not (item in Curr_Obs_1)]

    # if stepNo == 87:
    #     print(Curr_Obs)

    # print(Xe)
    if New_Targets:
        for ij in range(len(New_Targets)):
            Xe = np.vstack((Xe, np.dot(np.array(H).T, np.array(ordinaryFrame[New_Targets[ij], :]).T)))
            # print(np.array(P0).shape)
            Pe = np.vstack((Pe, np.array([P1])))

            # Ff[N_Target + ij -1] = [f, f]
            Term_Con.append(0)
    # print(np.array(Xe).shape,'\n')
    # print(Xe, '\n')
    # if stepNo == 87:
    #     print('12:', Xe.shape)
    return Xe.T, Pe, Term_Con, Terminated_objects_index


