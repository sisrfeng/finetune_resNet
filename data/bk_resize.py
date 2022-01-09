import cv2
import os
from os import path as op
from os import listdir as osl
#  -----------------------自动import结束------------------------#

opj = op.join
pwd = os.getcwd()
pwd = pwd + '/gls_train_val'
print('看pwd : ')
print(pwd)
for split in osl(pwd):
    for cls in osl(opj(pwd, split)):
        for img_path in os.listdir(opj(pwd,split,cls)):
            img_path = os.path.join(pwd,split,cls,img_path )
            img = cv2.imread(img_path)
            shape = img.shape
            print(f'处理前：{shape= }')
            if max(h,w) > 6000:
                img2 = cv2.resize(img, (shape[1]//8, shape[0]//8) )
                cv2.imwrite(img_path, img2)
