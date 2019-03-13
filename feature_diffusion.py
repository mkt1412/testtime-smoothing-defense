import numpy as np
import torch
from torchvision import models
from medpy.filter import anisotropic_diffusion
import torch.nn as nn
import foolbox
import util

def diffuse_features(features):
    '''
    diffuse intermediate features
    :param features: intermediate features
    :return: diffused features
    '''
    features = anisotropic_diffusion(features.cpu().numpy())
    features = (torch.from_numpy(features)).cuda()
    return features


def diffusion_and_forward(diff_list, model, image):
    '''
    implement anisotropic diffusion on intermediate layers
    :param diff_list: list of target layers
    :param model: pre-trained model
    :param image: input image(numpy)
    :return: prediction label
    '''
    image = torch.unsqueeze(torch.from_numpy(image), 0).cuda()

    modules = list(model.children())
    resnet_1st = modules[0:1]
    resnet_1st = nn.Sequential(*resnet_1st)
    for p in model.parameters():
        p.requires_grad = False
    features = resnet_1st(image)

    for i in range(1, len(modules)-1):
        layer = modules[i:i+1]
        layer = nn.Sequential(*layer)
        features = layer(features)
        if i in diff_list:
            features = diffuse_features(features)

    resnet_last = modules[-1:]
    resnet_last = nn.Sequential(*resnet_last)
    prob = resnet_last(features.view(features.size(0), -1))
    #prediction = np.argmax(prob.cpu().numpy())

    return prob.cpu().numpy()

img_path = "/home/chaotang/PycharmProjects/data/ILSVRC2012/val_correct/n01514668/ILSVRC2012_val_00023551.JPEG"
image = util.load_image(img_path=img_path, normalize=True)
resnet18 = models.resnet152(pretrained=True).cuda().eval()
diff_list = [1,2,3]
prediction = diffusion_and_forward(diff_list=diff_list, model=resnet18, image=image)

