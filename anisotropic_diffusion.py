import numpy as np
from util import load_image
from matplotlib import pyplot as plt
import warnings
from scipy.ndimage.filters import convolve
from medpy.filter import anisotropic_diffusion
import time


def anisotropic_diffusion_1d(input, niter=1, kappa=50, gamma=0.1, option=1):
    """
    Implementation of anisotropic diffusion on single-channel image
    :param input: input single-channel image
    :param niter: number of iterations
    :param kappa: factor k
    :param gamma: "learning rate" aka time step
    :param option: energy function option 1 or 2
    :return: diffused single-channel image
    """
    output = input.copy()

    for i in range(niter):

        Ix, Iy = np.gradient(output)
        Ixx = np.gradient(Ix, axis=0)
        Iyy = np.gradient(Iy, axis=1)
        if option == 1:
            C = 1 / (1 + (np.sqrt(Ix ** 2 + Iy ** 2) / kappa) ** 2)
        elif option == 2:
            C = np.exp(-((np.sqrt(Ix ** 2 + Iy ** 2) / kappa) ** 2))
        elif option == 3:
            C = 1 / 1 + (np.exp(np.sqrt(Ix**2 + Iy**2) - 20) * np.sqrt(Ix**2 + Iy**2)) ** 2
        else:
            warnings.warn("no such diffusion option")
            quit()
        Cx, Cy = np.gradient(C)
        It = Cx * Ix + Cy * Iy + C * (Ixx + Iyy)

        output = output + gamma * It

    return output


def anisotropic_diffusion_3d(input, niter=1, kappa=50, gamma=0.1, option=1):
    """
    Implementation of anisotropic diffusion on RGB image
    :param input: input RGB image
    :param niter: number of iterations
    :param kappa: factor k
    :param gamma: "learning rate" aka time step
    :param option: energy function option 1 or 2
    :return: diffused image
    """
    # TODO: add clip
    # TODO: plot.ion()
    if input.dtype == np.uint8:
        input = input / 255.0

    output = input.copy()

    for i in range(input.shape[0]):

        output[i] = anisotropic_diffusion_1d(input=input[i], niter=niter, kappa=kappa, gamma=gamma, option=option)

    output = output.clip(min=0.0, max=1.0)

    return output


if __name__ == "__main__":
    img_path = "/home/chao/PycharmProjects/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.05-0.01-10/n01685808/" \
               "ILSVRC2012_val_00016379.JPEG"
    raw_image = load_image(img_path=img_path, normalize=False, resize=False)
    plt.ion()
    for i in range(50):
        image = anisotropic_diffusion_3d(input=raw_image, niter=i, option=1, kappa=50, gamma=0.8)
        # image = anisotropic_diffusion(raw_image, niter=i, gamma=0.1, kappa=50)
        # # image = convolve(raw_image, weights=np.full((1, 3, 3), 1.0 / 9))
        image = np.transpose(image, (1, 2, 0))
        plt.figure(1)
        plt.title("step = {}".format(i))
        plt.imshow(image)
        plt.pause(0.5)
    plt.ioff()





