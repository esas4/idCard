import cv2
import os

readinfiles="/home/data/shizh/real_or_fake/idCard/2_test_real/"
readoutfiles="/home/data/shizh/real_or_fake/idCard/2_test_real_cropped/"
if not os.path.exists(readoutfiles):
    os.makedirs(readoutfiles)

pics=os.listdir(readinfiles)

for pic in pics:
    img=cv2.imread(readinfiles+pic)
    cropped = img[320:1900, 700:950]  # 号码
    # cropped=img[320:1900,950:1200] # 出生年月
    # cropped = img[320:1900,1200:1450] # 性别
    # cropped = img[320:1900, 1450:1700]  # 姓名
    cv2.imwrite(readoutfiles+pic,cropped)
    print('save '+pic)
    # quit()