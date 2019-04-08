import numpy as np
from util import load_image
from matplotlib import pyplot as plt
import warnings


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

        Ix, Iy = np.gradient(input)
        Ixx = np.gradient(Ix, axis=0)
        Iyy = np.gradient(Iy, axis=1)
        if option == 1:
            C = 1. / (1. + (np.sqrt(Ix ** 2 + Iy ** 2) / kappa) ** 2)
        elif option == 2:
            C = np.exp(-(np.sqrt(Ix**2 + Iy**2)/kappa**2))
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
    output = input.copy()

    for i in range(input.shape[0]):

        output[i] = anisotropic_diffusion_1d(input=input[i], niter=niter, kappa=kappa, gamma=gamma, option=option)

    return output


if __name__ == "__main__":
    img_path = "/home/chao/PycharmProjects/data/ILSVRC2012/" \
               "val_correct_adv_resnet152_pgd/n01514668/ILSVRC2012_val_00023551.JPEG"
    raw_image = load_image(img_path=img_path, normalize=False, resize=False)
    image = anisotropic_diffusion_3d(input=raw_image, niter=7, kappa=20, option=3)
    raw_image = np.transpose(raw_image, (1, 2, 0))
    image = np.transpose(image, (1, 2, 0))
    plt.figure("comparision")
    plt.subplot(1, 2, 1)
    plt.title("raw image")
    plt.imshow(raw_image)
    plt.subplot(1, 2, 2)
    plt.title("processed image")
    plt.imshow(image)
    plt.show()
