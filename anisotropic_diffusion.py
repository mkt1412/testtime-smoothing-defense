import numpy as np
from util import load_image
from matplotlib import pyplot as plt
import warnings
from cv2.ximgproc import anisotropicDiffusion


def gradient(input, axis=0, order=1):
    """
    Compute the gradient along certain axis
    :param input: input image
    :param axis: compute gradient along which axis
    :param order: first order gradient or second order
    :return:
    """

    output = input.copy()
    if order == 1:
        if axis == 1:
            output[:, 0] = input[:, 1] - input[:, 0]  # left boundary uses forward difference
            output[:, -1] = input[:, -1] - input[:, -2]  # right boundary uses backward difference
            output[:, 1:-1] = (input[:, 2:] - input[:, :-2])/2  # interior region uses central differences
        elif axis == 0:
            output[0, :] = input[1, :] - input[0, :]  # upper boundary uses forward difference
            output[-1, :] = input[-1, :] - input[-2, :]  # bottom boundary uses backward difference
            output[1:-1, :] = (input[2:, :] - input[:-2, :]) / 2  # interior region uses central differences
        else:
            print("axis does not exist")
            quit()

    elif order == 2:
        if axis == 0:
            output[0, :] = input[1, :] - input[0, :] / 2
            output[-1, :] = input[-2, :] - input[-1, :] / 2
            output[1:-1, :] = input[2:, :] - 2 * input[1:-1, :] + input[:-2, :]
        elif axis == 1:
            output[:, 0] = input[:, 1] - input[:, 0] / 2
            output[:, -1] = input[:, -2] - input[:, -1] / 2
            output[:, 1:-1] = input[:, 2:] - 2 * input[:, 1:-1] + input[:, :-2]
        else:
            print("axis does not exist")
            quit()

    return output


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

        Ix = gradient(output, axis=0, order=1)
        Iy = gradient(output, axis=1, order=1)
        Ixx = gradient(output, axis=0, order=2)
        Iyy = gradient(output, axis=1, order=2)

        # D denotes the diffusion coefficient
        if option == 1:
            D = 1 / (1 + (np.sqrt(Ix ** 2 + Iy ** 2) / kappa) ** 2)
        elif option == 2:
            D = np.exp(-((np.sqrt(Ix ** 2 + Iy ** 2) / kappa) ** 2))
        elif option == 3:
            D = 1 / 1 + np.sqrt(Ix ** 2 + Iy ** 2)
        else:
            warnings.warn("no such diffusion option")
            quit()

        Dx = gradient(D, axis=0)
        Dy = gradient(D, axis=1)
        It = Dx * Ix + Dy * Iy + D * (Ixx + Iyy)

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
        image = anisotropic_diffusion_3d(input=raw_image, niter=15, option=1, kappa=50, gamma=0.1)
    # image = anisotropicDiffusion(src=np.transpose((raw_image * 255).astype(np.uint8), (1, 2, 0)), niters=4, alpha=0.1,
    #                              K=20)
        image = np.transpose(image, (1, 2, 0))
        plt.figure(1)
        plt.title("step = {}".format(i))
        plt.imshow(image)
        # plt.show()
        plt.pause(0.5)
    plt.ioff()





