from torchvision import models, transforms
from PIL import Image
import cv2
import torch
from torchsummary import summary


transform = transforms.Compose(
    [
    # transforms.Resize(256),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
          mean=[0.485, 0.456, 0.406],  #  ImageNet的？
          std=[0.229, 0.224, 0.225] )
    ]
)


# print(dir(models))

img = Image.open("./boat/boat.jpeg")

img_t = transform(img)
in_batch = torch.unsqueeze(img_t, 0)

resnet = models.resnet18(pretrained=True)
#  summary(resnet, (3, 224,224))

# inference
resnet.eval()
preds = resnet(in_batch)
pred, class_idx = torch.max(preds, dim=1)
with open('imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
print(labels[class_idx])
