import numpy as np
import math
import time
import random


def Tree_Constructor(X, Sigma, F, Q, H, R, Z, S_l, d_G, pD, Beta, M, stepNo,no):
    X_p = X
    Sigma_p = Sigma

    # TMprob = []

    X_e = np.dot(np.mat(F), np.mat(np.array(X_p).reshape(4, 1)))
    # if stepNo == 99:
    #     print(no,Sigma_p)
    Sigma_e = Q + np.dot(np.dot(np.array(F), np.array(Sigma_p)), np.array(np.transpose(F)))
    S = np.dot(np.dot(np.mat(H), np.mat(Sigma_e)), np.mat(np.transpose(H))) + R
    K_E = np.dot((np.dot(np.mat(Sigma_e), np.mat(np.transpose(H)))), np.mat(S).I)
    X_E = X_e
    Sigma_E = Sigma_e
    S_E = S

    # if stepNo == 92:
    #     print(no, Sigma)

    # aaa = np.mat(Z[0]) - np.transpose(np.dot(np.mat(H), np.mat(X_e)))

    # if np.max(S) > S_l:
    #     S_G = S * S_l / np.max(S)
    # else:
    #     S_G = S

    S_G = S

    S_G = (S_G + np.transpose(S_G)) * 0.5
    eigValue, eigVector = np.linalg.eig(S_G)
    # if min(eigValue) <0:
    #     S_G = S_G + np.matrix([[-min(eigValue)+0.01,0],[0,-min(eigValue)+0.01]])
    # print(min(eigValue))
    if len(Z) == 0:
        TMindx = []
    else:
        num_meas = 0
        mahalanobisdis = []

        for index in Z:

            m_middle = (np.dot((Z[num_meas] - np.transpose(np.dot(H, X_e))), S_G.I))


            # print(np.dot(m_middle, (np.transpose(Z[num_meas] - np.transpose(np.dot(H, X_e))))),'\n')

            m_final = math.sqrt(np.dot(m_middle, (np.transpose(Z[num_meas] - np.transpose(np.dot(H, X_e))))))

            mahalanobisdis.append(m_final)
            num_meas += 1
        dis = sorted(mahalanobisdis)

        inx = np.argsort(mahalanobisdis)
        TMindx = [i for i, j in zip(inx, dis) if j < d_G]
    GMeaur = [Z[item] for item in TMindx]
    sizM = len(GMeaur)

    if len(TMindx) == 0:
        TMprob = (1 - pD) * Beta
    else:
        DS = [i for i in dis if i < d_G]
        aaa = (((2 * 3.1415926533) ** (M / 2)) * np.linalg.det(np.array(S)) ** 0.5)
        gij = math.e ** (- np.array(DS) * (np.array(DS) / 2)) / aaa

        TMprob = np.array([(1 - pD) * Beta], dtype=float)
        TMprob = np.vstack((TMprob.reshape(-1, 1), (gij * pD).reshape(-1, 1)))
    # print(TMindx, '\n')
    return TMindx, TMprob, X_E.reshape(4), Sigma_E, S_E, K_E

