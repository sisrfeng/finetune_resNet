# from __future__ import print_function
# from __future__ import division

import os
def find_gpus(num_of_cards_needed=6):
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >~/.tmp_free_gpus')
    # If there is no ~ in the path, return the path unchanged
    with open(os.path.expanduser ('~/.tmp_free_gpus'), 'r') as lines_txt:
        frees = lines_txt.readlines()
        idx_freeMemory_pair = [ (idx, int(x.split()[2]))
                                for idx, x in enumerate(frees) ]
    idx_freeMemory_pair.sort(reverse=True)  # 0号卡经常有人抢，让最后一张卡在下面的sort中优先
    idx_freeMemory_pair.sort(key=lambda my_tuple: my_tuple[1], reverse=True)
    usingGPUs = [str(idx_memory_pair[0]) for idx_memory_pair in
                    idx_freeMemory_pair[:num_of_cards_needed] ]
    usingGPUs = ','.join(usingGPUs)
    for pair in idx_freeMemory_pair[:num_of_cards_needed]:
        print('{}号: {} MB free'.format(*pair) )
    return usingGPUs


os.environ['CUDA_VISIBLE_DEVICES'] = find_gpus(1)  # 必须在import torch前面
import torch
# XPU: CPU or GPU
myXPU = torch.device('cuda')   # ('cuda:号数')   号数:从0到N, N是VISIBLE显卡的数量。号数默认是0 [不是显卡的真实编号]

#===================================保证可复现======================
an_int = 1
torch .manual_seed(an_int)
torch .cuda.manual_seed_all(an_int)
import numpy as np
np    .random.seed(an_int)
import random
random.seed(an_int)
torch .backends.cudnn.deterministic = True
torch .backends.cudnn.benchmark = False  # cuDNN supports many algorithms to compute convolution
                                        # autotuner runs a short benchmark and
                                        # selects the algorithm with the best performance
#===================================保证可复现======================

import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import copy


# the format of the directory 需要conforms to the ImageFolder structure
data_dir = "./data/small_gls_train_val/"
# data_dir = "./data/gls_train_val/"
#  data_dir = "./data/hymenoptera_data/"

# Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
model_name = "resnet"

# Number of classes in the dataset
num_classes = 6

# Batch size for training (change depending on how much memory you have)
batch_size = 16

# Number of epochs to train for
# num_epochs = 15
num_epochs = 20

freeze01 = True # only update the reshaped layer params
# When False, we finetune the whole model,


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
    since = time.time()

    val_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(myXPU)
                labels = labels.to(myXPU)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    # Special case for inception because in training it has an auxiliary output. In train
                    #   mode we calculate the loss by summing the final output and the auxiliary output
                    #   but in testing we only consider the final output.
                    if is_inception and phase == 'train':
                        # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                        outputs, aux_outputs = model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}' )

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_acc_history.append(epoch_acc)

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model, val_acc_history


def set_parameter_requires_grad(model, freeze01):
    if freeze01:
        for param in model.parameters():
            param.requires_grad = False


def initialize_model(model_name, num_classes, freeze01, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        """ Resnet18
        """
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze01)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze01)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg11_bn(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze01)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs,num_classes)
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze01)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze01)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, freeze01)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

# Initialize the model for this run
model_ft, input_size = initialize_model(model_name,
                                        num_classes,
                                        freeze01,
                                        use_pretrained=True)

# print(model_ft)


#  normalization for training
data_transforms = {
    'train': transforms.Compose([
                # Data augmentation
                transforms.RandomResizedCrop(input_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
    'val': transforms.Compose([
                transforms.Resize(input_size),
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ]),
            }


# Create  datasets
image_datasets = {x: datasets.ImageFolder( os.path.join(data_dir, x),  data_transforms[x])
                  for x in ['train', 'val']}

# Create dataloaders
dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x],
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=4)
                    for x in ['train', 'val']}


print("Params to learn:")
if freeze01:
    params_to_update = []
    for name,param in model_ft.named_parameters():
        if param.requires_grad :
            params_to_update.append(param)
            print("\t", name)
else:
    params_to_update = model_ft.parameters()
    for name,param in model_ft.named_parameters():
        if param.requires_grad :
            print("\t",name)

loss__ = nn.CrossEntropyLoss()

model_ft = model_ft.to(myXPU)
# Train and evaluate
model_ft, hist = train_model(model_ft,
                            dataloaders_dict,
                            loss__,
                            optim.SGD(params_to_update, lr=0.001, momentum=0.9),
                            num_epochs=num_epochs,
                            is_inception=(model_name=="inception")
                            )

# from scratch  vs  finetune
# Initialize the non-pretrained version of the model used for this run
model_scratch, _ = initialize_model(model_name,
                                    num_classes,
                                    freeze01=False,
                                    use_pretrained=False
                                    )

model_scratch = model_scratch.to(myXPU)
_,scratch_hist = train_model(model_scratch,
                            dataloaders_dict,
                            loss__,
                            optim.SGD(model_scratch.parameters(), lr=0.001, momentum=0.9),
                            # num_epochs=num_epochs,
                            num_epochs=num_epochs,
                            is_inception=(model_name=="inception"))

# Plot the training curves of validation accuracy vs. number
#  of training epochs for
# the transfer learning method
# v.s.
# the model trained from scratch

ft_hist = [h.cpu().numpy()
         for h in hist]
sc_hist = [h.cpu().numpy()
         for h in scratch_hist]

plt.title("Validation Accuracy vs. Training Epochs")
plt.xlabel("Training Epochs")
plt.ylabel("Validation Accuracy")
plt.plot(range(1,num_epochs+1),
         ft_hist,
         label="Pretrained")
plt.plot(range(1,num_epochs+1),
         sc_hist,
         label="Scratch")

plt.ylim((0,1.))
plt.xticks(np.arange(1, num_epochs+1, 1.0))
plt.legend()
# plt.show()
plt.savefig('./out/fig.png')