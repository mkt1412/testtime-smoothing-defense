import numpy as np
import torch
from torchvision import models
from medpy.filter import anisotropic_diffusion
import torch.nn as nn
import util
from torchsummary import summary
from matplotlib import pyplot as plt
from scipy.ndimage.filters import median_filter, convolve


def diffuse_features(features, mode="diffusion"):
    """
    Different diffusion mode
    :param features: intermediate features
    :param mode: diffusion mode
    :return: diffused features
    """
    features = features.cpu().numpy()  # convert features from tensor to array

    if mode == "diffusion":
        features = anisotropic_diffusion(features)
    elif mode == "mean":
        features = convolve(features, weights=np.full((1, 3, 3, 3), 1.0/27))
    elif mode == "median":
        features = median_filter(features, size=(1, 3, 3, 3))

    features = (torch.from_numpy(features)).cuda()  # from numpy back to tensor

    return features


def diffusion_and_forward(image, model, diff_list, visual_list, mode="diffusion"):
    """
    Implement diffusion and visualization on selected layers
    :param image: input image
    :param model: pre-trained nn model
    :param diff_list: diffusion layers
    :param visual_list: visualization layers
    :param mode: diffusion mode
    :return: probability
    """
    image = torch.unsqueeze(torch.from_numpy(image), 0).cuda()

    for p in model.parameters():
        p.requires_grad = False

    modules = list(model.children())
    resnet_1st = modules[0:1]
    resnet_1st = nn.Sequential(*resnet_1st)
    features = resnet_1st(image)

    plt.figure("features")
    plt.subplot(2,4,1)
    plt.title("layer0")
    plt.imshow(features[0, 20, :].cpu().numpy(), cmap="gray")

    for i in range(1, len(modules)-1):
        layer = modules[i:i+1]
        layer = nn.Sequential(*layer)
        features = layer(features)
        if i in diff_list:
            features = diffuse_features(features, mode=mode)
        if i in visual_list:
            plt.subplot(2, 4, i+1)
            plt.title("layer{}".format(i))
            plt.imshow(features[0, 20, :].cpu().numpy(), cmap="gray")
    plt.show()

    resnet_last = modules[-1:]
    resnet_last = nn.Sequential(*resnet_last)
    prob = resnet_last(features.view(features.size(0), -1))

    return prob.cpu().numpy()


if __name__ == "__main__":
    img_path = "/home/chaotang/PycharmProjects/data/ILSVRC2012/" \
               "val_correct_adv_resnet152_pgd/n01514668/ILSVRC2012_val_00023551.JPEG"
    image = util.load_image(img_path=img_path, normalize=True)
    resnet152 = models.resnet152(pretrained=True).cuda().eval()
    # summary(resnet152, (3, 224, 224))
    # print(resnet152)
    diff_list = [6]  # diffusion layers
    visual_list = [1, 2, 3, 4, 5, 6, 7]  # visualization layers
    prob = diffusion_and_forward(image=image, model=resnet152, diff_list=diff_list,
                                 visual_list=visual_list, mode="median")
    prediction = np.argmax(prob)
    print(prediction)


