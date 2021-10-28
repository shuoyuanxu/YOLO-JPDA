from yolo_pipeline import *
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from Initialization import initialization
from JPDA_Tracker import ApproxMultiscanJPDAProbabilities, JPDA
from JPDA_Tracker_ini import JPDA_ini
# import pickle

u_image = 448
v_image = 448

# Parameters
param = {}
# Parameters for Heuristics
param['Prun_Thre'] = 0.15  # Parameter for pruning detections with the confidence score less than this value
param['Term_Frame'] = 45  # The parameter for termination condition 45
param['tret'] = 15

# Parameters for Kalman Filtering and JPDA
q1 = 0.1  # The standard deviation of the process noise for the dynamic model
qm = 7.00  # The standard deviation of the measurement noise
param['Vmax'] = 7  # maximum velocity that a target can has (used for initialization only)

param['PD'] = 0.95  # Detection Probabilty or the average of true positive rate for detections
param['Beta'] = 3 / (u_image * v_image)  # Beta is False detection (clutter) density (Poisson assumption)
param['Gate'] = 30 ** 0.5  # Gate size for gating
param['S_limit'] = 100  # parameter to stop the gate size growing too much
# param['N_H'] = 5.4772  # Number of m-best solutions for approximating JPDA distribution

# Tracking Model
model = {}
model['T'] = 1  # Temporal sampling rate

# Dynamic model
F11 = [[1, model['T']], [0, 1]]
model['F'] = [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1],
              [0, 0, 0, 1]]  # The transition matrix for the dynamic model, from F11
Q11x = [[model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1], [model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1]]
Q11y = [[model['T'] ** 4 / 4 * q1, model['T'] ** 3 / 2 * q1], [model['T'] ** 3 / 2 * q1, model['T'] ** 2 * q1]]
# blkdiag(Q11x,Q11y) The process covariance matrix for the dynamic model 1
model['Q'] = [[0.025, 0.05, 0, 0], [0.05, 0.1, 0, 0], [0, 0, 0.025, 0.05], [0, 0, 0.05, 0.1]]

# Measurement model
model['H'] = [[1, 0, 0, 0], [0, 0, 1, 0]]  # Measurement matrix
model['R'] = qm * np.eye(2)  # Measurement covariance matrix [1,0,0,1]

# def load_calibration(calib_file):
#     with open(calib_file, 'rb') as file:
#         # print('load calibration data')
#         data = pickle.load(file)
#         mtx = data['mtx']       # calibration matrix
#         dist = data['dist']     # distortion coefficients
#
#     return mtx, dist
#
#
# calib_file = 'calibration_pickle.p'
# mtx, dist = load_calibration(calib_file)


def pipeline_yolo(img, n_frame):
    # img_undist = cv2.undistort(img, mtx, dist, None, mtx)
    output, detection_data = vehicle_detection_yolo(img, n_frame)
    return output, detection_data


if __name__ == "__main__":

    # first frame
    i = 519
    filename = 'examples/demo/a' + str(i) + '.jpg'
    # filename = 'examples/demo/qq.jpg'
    image1 = mpimg.imread(filename)
    yolo_result, detection_data1 = pipeline_yolo(image1, i)
    detec_d = np.array(detection_data1)
    # print(np.transpose(detection_data1))
    for j in range(len(detection_data1)):
        x = int(detection_data1[j][1])
        y = int(detection_data1[j][2])
        w = int(detection_data1[j][3]) // 2
        h = int(detection_data1[j][4]) // 2
        cv2.rectangle(image1, (x - w, y - h), (x + w, y + h), (0, 0, 255), 4)
        cv2.rectangle(image1, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
    cv2.imwrite("jpda/" + str(i) + ".jpg", image1)

    # second frames
    i = 520
    filename = 'examples/demo/a' + str(i) + '.jpg'
    image = mpimg.imread(filename)
    yolo_result, detection_data2 = pipeline_yolo(image, i)

    # Initialization
    # ' The distribution parameters for initial state p(x_0)
    model['X0'] = initialization(np.transpose(detection_data1), np.transpose(detection_data2), param, model)  # The initial mean
    model['P0'] = [[qm, 0, 0, 0], [0, 1, 0, 0], [0, 0, qm, 0], [0, 0, 0, 1]]  # The initial covariance
    # print(model['P0'])
    # print(model[''])
    Terminated_objects_index1 = []
    Xe1, Pe1, Term_Con1 = JPDA_ini('./Frame1.csv', model, param)

    # later frames
    i = 520
    for item in range(1048):
        filename = 'examples/demo/a' + str(i) + '.jpg'
        # print(filename)
        image = mpimg.imread(filename)
        yolo_result, detection_data = pipeline_yolo(image, i)
        print(Xe1)
        Xe, Pe, Term_Con, Terminated_objects_index = JPDA(np.transpose(detection_data), model, param, Xe1, Pe1, Term_Con1, item,
                                                          Terminated_objects_index1)
        Xe1 = Xe
        Pe1 = Pe
        Term_Con1 = Term_Con
        Terminated_objects_index1 = Terminated_objects_index
        j = 0
        # print(detection_data)
        # print(Xe1[0])
        if not(len(detection_data)):
            cv2.imwrite("jpda/" + str(i) + ".jpg", image)
        else:
            for item in Xe1[0]:
                x = int(Xe1[0][j])
                y = int(Xe1[2][j])
                w = int(detection_data[0][3]) // 2
                h = int(detection_data[0][4]) // 2
                cv2.rectangle(image, (x - w, y - h), (x + w, y + h), (0, 0, 255), 4)
                cv2.rectangle(image, (x - w, y - h - 20), (x + w, y - h), (125, 125, 125), -1)
                j += 1
            cv2.imwrite("jpda/" + str(i) + ".jpg", image)
        i += 1

    # plt.figure()
    # plt.imshow(yolo_result)
    # plt.title('yolo pipeline', fontsize=30)
    # plt.show()
