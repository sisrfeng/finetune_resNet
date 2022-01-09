import torch
import torch.nn as nn
from torchvision import models
from torch.hub import load_state_dict_from_url

import cv2
import numpy as np

from torchvision import transforms


with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

ori_image = cv2.imread('./human/human2.jpeg')
image = cv2.cvtColor(ori_image, cv2.COLOR_BGR2RGB)

transform = transforms.Compose([
             transforms.ToTensor(),
             transforms.Normalize(
                         mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]     )
             ])

image = transform(image)  # shape: (3,H,W)
image = image.unsqueeze(0)  # shape: (1,3,H,W)


class Full_Conv_ResNet(models.ResNet):
    # Start with standard resnet18
    def __init__(self, poolSize, num_classes=1000, pretrained=False, **kwargs):

        super().__init__(block=models.resnet.BasicBlock,
                        layers=[2, 2, 2, 2],
                        num_classes=num_classes,
                        **kwargs)
        #  4 Convolutional blocks
        # Each block contains 4 convolutional layers.
        if pretrained:
            state_dict = load_state_dict_from_url(models.resnet.model_urls["resnet18"], progress=True)
            self.load_state_dict(state_dict)

        # pytorch让resNet接收任意size的办法：AdaptiveAvgPool2d
        #    collapses the feature maps of any size to the predefined one.
        self.avgpool = nn.AvgPool2d(poolSize)  # Replace AdaptiveAvgPool2d with standard AvgPool2d

        # 原resnet的FC变成conv
        self.conv_as_FC = torch.nn.Conv2d(self.fc.in_features,
                                         num_classes,
                                         kernel_size=1)
        #  FC的参数塞给conv
        fc_w = self.fc.weight.data
        self.conv_as_FC.weight.data.copy_(fc_w.view(*fc_w.shape, 1, 1))
        self.conv_as_FC.  bias.data.copy_(self.fc.bias.data)

    def _forward_impl(self, x):
        # Standard forward for resnet18
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)

        # We forward pass through the last conv layer Instead of
        # through the original fully connected layer.
        x = self.conv_as_FC(x)
        return x
# if __name__ == "__main__":


for pool_h in range(7,15,2):
    for pool_w in range(7,15,2):
        print()
        print()
        try:
            pool_size0, pool_size1 = pool_h, pool_w
            poolSize = (pool_size0, pool_size1)

            # Load modified resnet18 model with pretrained ImageNet weights
            model = Full_Conv_ResNet(poolSize, pretrained=True).eval()

            #  inference.
            with torch.no_grad():

                #  Instead of a 1(batch_size) x1000 vector, we will get a  1x1000xnxm output.
                #  for each 1000 class, 有nxm的像素, 各代表一个区域是该class的概率值
                #  n,m 约等于 h/240, w/240  相当于裁成nxm张图送进resnet，但免去了重复提特征？
                infer_N_C_h_w = model(image)

                infer_N_C_h_w = torch.softmax(infer_N_C_h_w, dim=1)


                # Find the class with the maximum score in the n x m output map
                max_prob_map, class_idx = torch.max(infer_N_C_h_w, dim=1)

                row_max, row_idx = torch.max(max_prob_map, dim=1) #  max_prob_map.shape:  (1,3,8)
                col_max, col_idx = torch.max(row_max, dim=1)
                # print(f"  {class_idx[:]=}   ")
                pred_cls_id = class_idx[0,
                                        row_idx[0, col_idx],
                                        col_idx]

                # Print top predicted class
                print(f'类别名："{labels[pred_cls_id]}" ' )
                print('cls_id：', np.asarray(pred_cls_id))

                # Find the n x m score map for the predicted class
                print('Response map shape : ', np.asarray(infer_N_C_h_w.shape))
                score_map = infer_N_C_h_w[0, pred_cls_id, :, :].cpu().numpy()
                score_map = score_map[0]

                wh        = ori_image.shape
                score_map = cv2.resize(score_map,  (wh[1],  wh[0]))

                # Binarize score map
                _, map4contours = cv2.threshold(score_map,
                                              0.25,
                                              1,
                                              type=cv2.THRESH_BINARY)
                map4contours = map4contours.astype(np.uint8).copy()


                # Apply score map as a mask to original image
                score_map = score_map - np.min(score_map[:])
                score_map = score_map / np.max(score_map[:])  #  0到1

                score_map = cv2.cvtColor(score_map, cv2.COLOR_GRAY2BGR)
                masked_image = (ori_image * score_map).astype(np.uint8)


                # cv2.imwrite(f"score_map_{kernelSize}.png", score_map*255)
                cv2.imwrite(f"{pool_h}_{pool_w}_{str(infer_N_C_h_w.shape)[19:-1]}.png", masked_image)
                print('cv2.imwrite.   done')
                # cv2.imwrite("Original Image.png", original_image)
                # cv2.imshow("归一化后的score_map.png", score_map)
                # key = cv2.waitKey(0)
                # if key == ord('q'):
                    # exit()

        except Exception as e:
            print(f'  {e=}   ')
            break
