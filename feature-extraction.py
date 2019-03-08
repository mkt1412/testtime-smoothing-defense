"""
Forward pass by block for pretrained resnet18
"""

from torchvision import models
import foolbox
import torch
import torch.nn as nn
import numpy as np


# initiate the model
resnet18 = models.resnet18(pretrained=True).cuda().eval()
#summary(resnet18, (3, 224, 224))
mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
model = foolbox.models.PyTorchModel(resnet18, bounds=(0,1), num_classes=1000, preprocessing=(mean, std))

# get source image and label
image, label = foolbox.utils.imagenet_example(data_format="channels_first")
image = image/255

# apply attack on source image
attack = foolbox.attacks.GradientSignAttack(model)
adv_image = attack(image, label, max_epsilon=0.5, epsilons=1)

print("adv predicted class:", np.argmax(model.predictions(adv_image)), "true class:", label)

# convert original image to tensor
image = torch.from_numpy(image)
image = torch.unsqueeze(image, 0)
image = image.cuda()

# convert attacked image to tensor
adv_image = torch.from_numpy(adv_image)
adv_image = torch.unsqueeze(adv_image, 0)
adv_image = adv_image.cuda()

# predict label with original resnet18
with torch.no_grad():
    out_1 = np.argmax(resnet18(image).cpu())
    out_2 = np.argmax(resnet18(adv_image).cpu())
print(out_1, out_2)

# resnet part 1: from layer 0 to layer 2
modules = list(resnet18.children())[:3]
resnet_1st = nn.Sequential(*modules)
for p in resnet18.parameters():
    p.requires_grad = False

# resnet part 2: from layer 2 to layer -2
modules = list(resnet18.children())[3:-1]
resnet_2nd = nn.Sequential(*modules)
for p in resnet18.parameters():
    p.requires_grad = False

# resnet part 3: fully connected layer
modules = list(resnet18.children())[-1:]
resnet_3rd = nn.Sequential(*modules)
for p in resnet18.parameters():
    p.requires_grad = False

out_1st = resnet_1st(image)
out_2nd = resnet_2nd(out_1st)
out_3rd = resnet_3rd(out_2nd.view(out_2nd.size(0), -1))
print(np.argmax(out_3rd.cpu().numpy()))
















