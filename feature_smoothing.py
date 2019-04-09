import numpy as np
import torch
from torchvision import models
from medpy.filter import anisotropic_diffusion
from cv2.ximgproc import anisotropicDiffusion
from cv2 import bilateralFilter
import torch.nn as nn
import util
from util import load_image, load_pkl_image
from torchsummary import summary
from matplotlib import pyplot as plt
from scipy.ndimage.filters import median_filter, convolve, gaussian_filter
import getpass
import time
from modified_curvature_motion import modified_curvature_motion
import warnings


def smooth(inputs, mode="anisotropic-diffusion", param=None):
    """
    Different diffusion mode
    :param inputs: input of a layer that needs to be smoothed, either raw images (numpy) or inter-features (tensor)
                   (note: "input" is a reserved key word in python)
    :param mode: smoothing method
    :param param: parameters of the smoothing method
    :return: smoothed input
    """
    is_feature = type(inputs) != np.ndarray  # whether the input is an intermediate feature (tensor)
    if is_feature:
        inputs = inputs.cpu().numpy()  # convert features from tensor to array

    if mode == "anisotropic-diffusion":
        # TODO: need to take care of the two sets of arguments on iteration number, medpy need around 10, cv2 need 4
        param = (0.1, 20, 4) if param is None else param
        if is_feature:
            inputs = anisotropic_diffusion(inputs, kappa=param[1], niter=int(param[2]))
        else:
            inputs = anisotropicDiffusion(np.transpose((inputs * 255).astype(np.uint8), (1, 2, 0)),
                                          alpha=param[0], K=param[1], niters=int(param[2]))
            inputs = np.transpose(inputs, (2, 0, 1)).astype("float32") / 255
    elif mode == "mean":
        param = (1, 1, 3, 3) if param is None else param
        if is_feature:
            inputs = convolve(inputs, weights=np.full(param, 1.0/27))
        else:
            inputs = convolve(inputs, weights=np.full(param[1:], 1.0/9))
            inputs = ((inputs * 255).astype(np.uint8) / 255.0).astype(np.float32)
    elif mode == "median":
        param = (1, 1, 3, 3) if param is None else param
        if is_feature:
            inputs = median_filter(inputs, size=param)
        else:
            inputs = median_filter(inputs, size=param[1:])
            inputs = ((inputs * 255).astype(np.uint8) / 255.0).astype(np.float32)
    elif mode == "modified-curvature-motion":
        if is_feature:
            param = (1, 0.1) if param is None else param
            inputs = inputs.squeeze()
            inputs = modified_curvature_motion(inputs, niter=int(param[0]), k=param[1])
            inputs = np.expand_dims(inputs, axis=0)
        else:
            param = (20, 0.9) if param is None else param
            inputs = modified_curvature_motion(inputs, niter=int(param[0]), k=param[1])
            inputs = ((inputs * 255).astype(np.uint8) / 255.0).astype(np.float32)
    elif mode == "bilateral":
        if not is_feature:  # only works for raw images, [0, 1] works better than [0, 255]
            param = (9, 75, 75) if param is None else param
            inputs = np.transpose(bilateralFilter(np.transpose(inputs, (1, 2, 0)),
                                                  d=int(param[0]),
                                                  sigmaColor=(param[1]),
                                                  sigmaSpace=(param[2])),
                                  (2, 0, 1))
    elif mode == "gaussian":
        param = (3,) if param is None else param
        inputs = gaussian_filter(inputs, param[0]).astype("float32")

    elif mode == 'quantization':
        inputs = ((inputs * 255).astype(np.uint8) / 255.0).astype(np.float32)

    elif mode == "customized":
        pass
    else:
        warnings.warn("no such smoothing method")
        quit()

    if is_feature:
        inputs = (torch.from_numpy(inputs)).cuda()  # convert numpy back to tensor, for features only

    return inputs


def forward_and_smooth(image, model, smooth_list=[], visual_list=[], mode="anisotropic-diffusion", param=None):
    """
    Layer by layer operation for forward and smooth
    :param image: input image, numpy array with shape (3 x 224 x 224)
    :param model: pre-trained nn model
    :param smooth_list: list of layers whose input shall be smoothed
    :param visual_list: list of layers whose feature maps will be visualized
    :param mode: smoothing mode
    :param visual_flag: visualization flag
    :param param: parameters of the smoothing method
    :return: probability
    """
    if -1 in smooth_list:
        image = smooth(image, mode=mode, param=param)

    # Normalize and convert to tensor
    image = torch.unsqueeze(util.normalize_image_to_tensor(image), 0).cuda()

    for p in model.parameters():
        p.requires_grad = False

    modules = list(model.children())

    if len(visual_list) != 0:
        plt.figure("features")

    features = 0

    for i in range(0, len(modules)-1):
        layer = modules[i:i+1]
        layer = nn.Sequential(*layer)
        if i == 0:
            features = layer(image)
        else:
            features = layer(features)
        if i in smooth_list:
            features = smooth(features, mode=mode, param=param)
        if i in visual_list and len(visual_list) != 0:
            plt.subplot(2, 4, i+1)
            plt.title("layer{}".format(i))
            plt.imshow(features[0, 20, :].cpu().numpy(), cmap="gray")
    if len(visual_list) != 0:
        plt.show()

    resnet_last = modules[-1:]
    resnet_last = nn.Sequential(*resnet_last)
    prob = resnet_last(features.view(features.size(0), -1))

    return prob.cpu().numpy()


if __name__ == "__main__":
    # All directories must end with '/'
    if getpass.getuser() == 'fantasie':  # user is Yifei
        ROOT_DIR = '/media/fantasie/backup/data/ILSVRC2012/'  # root directory for ImageNet
    else:  # user is Chao
        ROOT_DIR = '/home/chao/PycharmProjects/data/ILSVRC2012/'

    img_path = ROOT_DIR + 'val_correct_adv_resnet152_pgd-0.01-0.002/n01440764/ILSVRC2012_val_00000293.JPEG'
    if img_path.endswith('pkl'):
        image = load_pkl_image(img_path)
    else:
        if 'resnet152' in img_path:  # adversarial images, already resized
            image = load_image(img_path, resize=False)
        else:  # clean images, need resizing
            image = load_image(img_path, resize=True)

    resnet152 = models.resnet152(pretrained=True).cuda().eval()
    # summary(resnet152, (3, 224, 224))
    # print(resnet152)

    smooth_list = [-1, 4]  # diffusion layers
    visual_list = list(range(8))  # visualization layers

    start = time.time()
    prob = forward_and_smooth(image=image, model=resnet152, smooth_list=smooth_list,
                              visual_list=visual_list, mode="modified-curvature-motion")
    print("Forward and smooth time: ", time.time() - start)

    prediction = np.argmax(prob)
    print(prediction)


