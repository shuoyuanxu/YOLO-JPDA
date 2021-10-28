import numpy as np
import time
import scipy
import random


def JPDA_Probabilty_Calculator(M):
    N_T = len(M)
    F_Pr = []
    PT = np.zeros((1, N_T), dtype=float)
    msp = M[0]['Meas_edge'][1, -1]
    Hypo_indx = np.asarray([len(item['Prob']) - 1 for item in M])
    for i in range(N_T):
        ind0 = [item for item in range(len(M[i]['Meas_edge'][1])) if M[i]['Meas_edge'][1][item] == 1]
        F_Pr.append(np.zeros((len(ind0), 1), dtype=float))

    if N_T == 1:
        P_T = np.prod(M[0]['Prob'], 1)
        for kk in range(len(ind0)):
            temp = [P_T[item] for item in range(len(M[0]['Hypo'][:, 0])) if
                    M[0]['Hypo'][item, 0] == M[-1]['Meas_edge'][0, ind0[kk]]]
            F_Pr[-1][ind0[kk], 0] = sum(temp)
    else:
        a = np.zeros((1, N_T), dtype=int)
        a[0, -1] = -1
        temp = np.zeros((1, N_T), dtype=int)
        temp[0, -1] = 1
        t = 0

        while max(np.abs(a[0] - Hypo_indx)) > 1e-3 and (t < 1000):
            a[0] += temp[0]
            hypothesis = np.zeros((msp, N_T))
            for j in range(N_T - 1, -1, -1):
                if a[0][j] > Hypo_indx[j]:
                    a[0][j] = 0
                    a[0][j - 1] += 1
                PT[0, j] = np.prod(M[j]['Prob'][a[0][j], :])

                hypothesis[:, j] = M[j]['Hypo'][a[0][j], :].T

            chkk = 0
            for jj in range(msp):
                zhpo = [item for item in range(len(hypothesis[jj])) if hypothesis[jj, item] == 0]
                if ((zhpo == []) and (len(np.unique(hypothesis[jj])) == N_T)) or (
                    len(np.unique(hypothesis[jj])) == N_T - len(zhpo) + 1):
                    chkk += 1
                else:
                    break
            if chkk == msp:
                for i in range(N_T):
                    indd = [item for item in range(len(M[i]['Meas_edge'][1]))
                            if
                            M[i]['Meas_edge'][1, item] == 1 and M[i]['Meas_edge'][0, item] == M[i]['Hypo'][a[0][i], 0]]
                    for itemTemp in indd:
                        F_Pr[i][itemTemp][0] += np.prod(PT)
            t += 1

    for item in range(len(F_Pr)):
        F_Pr[item] = F_Pr[item] / sum(F_Pr[item])
    return F_Pr


if __name__ == '__main__':
    mbest = 100

    Mes_Tar = np.zeros((36, 36), dtype=float)
    Mes_Tar[21, 0] = 1
    Mes_Tar[21, 1] = 1
    Mes_Tar[25, 2] = 1
    Mes_Tar[23, 3] = 1
    Mes_Tar[27, 4] = 1
    Mes_Tar[22, 5] = 1
    Mes_Tar[28, 6] = 1
    Mes_Tar[20, 7] = 1
    Mes_Tar[19, 8] = 1

    Mes_Tar[26, 9] = 1
    Mes_Tar[29, 9] = 1
    Mes_Tar[24, 10] = 1
    Mes_Tar[35, 11] = 1
    Mes_Tar[24, 12] = 1
    Mes_Tar[24, 13] = 1
    Mes_Tar[33, 14] = 1
    Mes_Tar[24, 16] = 1

    obj_info = []
    mass_1 = {'Meas_edge': np.array([[0, 3], [1, 1]]),
              'Hypo': np.array([[0], [3]]),
              'Prob': np.array([[3.39e-7], [0.0012]]),
              'A_Eq_const': [1, 1],
              'B_Eq_const': 1,
              'Cost': np.array([[14.897], [6.7287]])}

    mass_2 = {'Meas_edge': np.array([[0, 3], [1, 1]]),
              'Hypo': np.array([[0], [3]]),
              'Prob': np.array([[3.39e-7], [0.0101]]),
              'A_Eq_const': [1, 1],
              'B_Eq_const': 1,
              'Cost': np.array([[14.897], [4.5989]])}
    for i in range(19):
        if i == 0:
            obj_info.append(mass_1)
        else:
            obj_info.append(mass_2)
            # ApproxMultiscanJPDAProbabilities(Mes_Tar,obj_info,mbest)
