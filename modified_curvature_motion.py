import numpy as np
from util import load_image
from matplotlib import pyplot as plt


def modified_curvature_motion(image, niter=1, k=0.05):
    """
    Implementation of Modified Curvature Motion
    :param image: input image
    :param niter: number of iterations
    :param k: factor
    :return: smoothed and enhanced image, values in [0, 1]
    """
    # The method works better for normalized images with intensity values in [0, 1]
    normalized = np.max(image) <= 1.0
    if not normalized:
        image = image / 255.0

    for i in range(niter):
        # compute gradients
        # compute gradients
        Ix, Iy, Iz = np.gradient(image)
        Ixx, Ixy, Ixz = np.gradient(Ix)
        _, Iyy, Iyz = np.gradient(Iy)
        Izz = np.gradient(Iz, axis=2)

        Ix2 = Ix ** 2
        Iy2 = Iy ** 2
        Iz2 = Iz ** 2

        part1 = 1 / (k ** 2) * (Ixx + Iyy + Izz)
        part2 = (Iy2 + Iz2) * Ixx + (Ix2 + Iz2) * Iyy + (Ix2 + Iy2) * Izz
        part3 = -2 * (Ix * Iy * Ixy + Ix * Iz * Ixz + Iy * Iz * Iyz)
        part4 = (1 / (k ** 2) + Ix2 + Iy2 + Iz2) ** 2

        It = (part1+part2+part3)/part4

        image = image + 0.1 * It
        image = image.clip(min=0.0, max=1.0)  # intensity should be in the valid domain

    return image


if __name__ == "__main__":
    img_path = "/home/chao/PycharmProjects/data/ILSVRC2012/" \
               "val_correct_adv_resnet152_pgd/n01514668/ILSVRC2012_val_00023551.JPEG"
    raw_image = load_image(img_path=img_path, normalize=False, resize=False)
    image = modified_curvature_motion(image=raw_image, niter=30, k=0.1)
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
