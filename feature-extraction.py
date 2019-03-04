from torchvision import models
import foolbox
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt
import medpy.filter as filter
import matplotlib.image as mping



# initiate the model
resnet18 = models.resnet18(pretrained=True).cuda().eval()
mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
model = foolbox.models.PyTorchModel(resnet18, bounds=(0,1), num_classes=1000, preprocessing=(mean, std))
#print(resnet18)
#print(model)


# get source image and label
image, label = foolbox.utils.imagenet_example(data_format="channels_first")
#print(label)
image = image/255
#print("predicted class:", np.argmax(model.predictions(image)), "true class:", label)
#print(np.shape(image))

# apply attack on source image
attack = foolbox.attacks.GradientSignAttack(model)
#print(type(attack))
adv_image = attack(image, label, max_epsilon=0.5, epsilons=1)


print("adv predicted class:", np.argmax(model.predictions(adv_image)), "true class:", label)


# set original and adversarial image
adv_image = torch.from_numpy(adv_image)
adv_image = torch.unsqueeze(adv_image, 0)
adv_image = adv_image.cuda()

ori_image = torch.from_numpy(image)
ori_image = torch.unsqueeze(ori_image, 0)
ori_image = ori_image.cuda()

# remove some layers
modules = list(resnet18.children())[:-2]
resnet18 = nn.Sequential(*modules)
for p in resnet18.parameters():
    p.requires_grad = False
#print(resnet18)


# output feature in gpu tensor
out_adv = resnet18(adv_image)
out_ori = resnet18(ori_image)

#print(out_adv.shape)


out_adv = torch.squeeze(out_adv, 0)
out_ori = torch.squeeze(out_ori, 0)

# output feature in array
adv_feature = out_adv[10].cpu().numpy()
ori_feature = out_ori[10].cpu().numpy()

# original and adversarial image in array
adv_image_np = np.transpose(torch.squeeze(adv_image.cpu(), 0).numpy(), (1,2,0))
ori_image_np = np.transpose(image, (1,2,0))

# plotting
plt.figure()
plt.subplot(1,4,2)
plt.imshow(ori_feature, cmap="gray")
plt.subplot(1,4,1)
plt.imshow(ori_image_np)
plt.subplot(1,4,4)
plt.imshow(adv_feature ,cmap="gray")
plt.subplot(1,4,3)
plt.imshow(adv_image_np)
plt.show()

#print(ori_feature-adv_feature)














