import cv2
import numpy as np
# a = 7022
# b = 7022
# for i in range(0, 179):
#     str1 = str(a)
#     str2 = str(b)
#     gname = "C:/Users/s212067/Dropbox/CODES/image_processing/Raspberry Code/cvpr10_tud_stadtmitte/DaMultiview-seq" + str1 + ".png"
#     downname = "C:/Users/s212067/Dropbox/CODES/image_processing/Raspberry Code/cvpr10_tud_stadtmitte/DaMultiview-seq" + str2 + ".jpg"
#     img = cv2.imread(gname)
#     imc = cv2.resize(img, (640, 480))
#     cv2.imwrite(downname, imc)
#     a += 1
#     b += 1
#     # chopped
#     # img = cv2.imread(gname, 0)
#     # cropped = img[0:480, 80:560]
#     # cv2.imwrite(downname, cropped)
#     # a += 1
#     # b += 1

fps = 25  # 视频帧率
fourcc = cv2.VideoWriter_fourcc(*'MJPG')
videoWriter = cv2.VideoWriter('jpda.wmv', fourcc, fps, (896, 448))
for i in range(0, 1049):
    p1 = 1
    p2 = i
    img12 = cv2.imread('jpda/' + str(p1+p2) + '.jpg')
    img22 = cv2.imread('demo/' + str(p1+p2) + '.jpg')
    cv2.putText(img12, "JPDA", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    cv2.putText(img22, "YOLO", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, 255)
    vis = np.zeros((448, 896, 3), np.uint8)
    vis[:448, :448, :3] = img12
    vis[:448, 448:896, :3] = img22
    videoWriter.write(vis)
# videoWriter.release()

