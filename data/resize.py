import cv2
import os
from os import path as op
from os import listdir as osl
#  -----------------------自动import结束------------------------#

opj = op.join
pwd = os.getcwd()
root = pwd + '/gls_train_val'
print('看root : ')
print(root)
for split in osl(root):
    for cls in osl(opj(root, split)):
        for img_n in osl(opj(root, split, cls)):
            img_path = opj(root, split, cls, img_n)
            img = cv2.imread(img_path)
            shape = img.shape
            print(f'处理前:{shape= }')
            if max(shape[:]) > 6000:
                img2 = cv2.resize(img, (shape[1]//8, shape[0]//8))
                out_path = opj(pwd, 'small_gls_train_val', split, cls)
                if not op.exists(out_path):
                    os.system(f'mkdir -p {out_path}')
                cv2.imwrite(f'{out_path}/{img_n}', img2)
