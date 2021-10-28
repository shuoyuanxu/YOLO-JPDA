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


def ismember(a, b):
    if a in b:
        return 0
    else:
        return 1


def cellfundevidedsum(b):
    if type(b) == float:
        c = 1
    else:
        c = [item / sum(b) for item in b]
    return c


def JPDA_ini(path, model, param, Image_address=None):
    # check the number of input arguments e.g .3 <= n_input <= 4
    detection_new = np.loadtxt(path, dtype=float, delimiter=",", skiprows=0)

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

    # The initial state's distribution models
    X0 = model['X0']  # The initial mean
    P0 = model['P0']  # The initial covariance

    # JPDA Parameters
    PD = param['PD']  # Detection Probability
    Beta = param['Beta']  # False detection (clutter) likelihood
    Gate = param['Gate']  # Gate size for gating
    S_limit = param['S_limit']  # parameter to stop the gate size growing too much
    # mbest = param['N_H']  # Threshold for considering m-best number of Hypotheses for JPDA

    DMV = len(H)
    DSV = len(H[0])

    N_Target = len(X0)

    # ---------------------------------- Initial Parameters -------------------------------------
    # Initial State Vector
    Xe = []
    Pe = []
    BxT = []
    Ff = []
    Term_Con = []

    ij = 0
    for index in list(range(N_Target)):
        Ff.append([1, 1])
        Term_Con.append(0)
        Xe.append(list(np.ndarray.flatten(np.array(X0[ij]))))
        Pe.append(P0)
        ij += 1

    Xe = list(np.transpose(Xe))
    print('adasdasdasd', Xe, 'adasdasd', Pe)
    # ----------------------- loading Image and detections for first frame -----------------------
    return Xe, Pe, Term_Con
    # print(Xe)

