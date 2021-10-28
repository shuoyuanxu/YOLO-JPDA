import cv2

a = 1
b = 1
for i in range(1, 10710):
    str1 = str(a)
    str2 = str(b)
    gname = "examples/demo/a" + str1 + ".jpg"
    downname = "examples/demo/a" + str2 + ".jpg"
    img = cv2.imread(gname)
    abc = cv2.resize(img, (448, 448))
    cv2.imwrite(downname, abc)
    a += 1
    b += 1
    # chopped
    # img = cv2.imread(gname, 0)
    # cropped = img[0:480, 80:560]
    # cv2.imwrite(downname, cropped)
    # a += 1
    # b += 1
