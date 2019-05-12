import numpy as np
from util import load_image
from matplotlib import pyplot as plt


def shift_image(img, shift, axis=None):
    """
    Shift the image according to the specified stepsize and pad it with edge pixels
    """
    out = np.roll(img, shift, axis=axis)
    if shift == 1:
        if axis == 0:
            out[0, :, :] = out[1, :, :]
        elif axis == 1:
            out[:, 0, :] = out[:, 1, :]
        else:
            out[:, :, 0] = out[:, :, 1]
    elif shift == -1:
        if axis == 0:
            out[-1, :, :] = out[-2, :, :]
        elif axis == 1:
            out[:, -1, :] = out[:, -2, :]
        else:
            out[:, :, -1] = out[:, :, -2]
    else:
        return img

    return out


def modified_curvature_motion(image, niter=1, k=0.05):
    """
    Implementation of Modified Curvature Motion
    :param image: input image, numpy array with size (3 x h x w)
    :param niter: number of iterations
    :param k: factor
    :return: smoothed and enhanced image, values in [0, 1]
    """
    # The method works better for normalized images with intensity values in [0, 1]
    if image.dtype == np.uint8:
        image = image / 255.0

    for i in range(niter):
        # Compute first-order gradients
        Ix, Iy, Iz = np.gradient(image)

        # Compute second-order derivatives
        # Reference: http://www.mathematik.uni-dortmund.de/~kuzmin/cfdintro/lecture4.pdf
        Ipx, Imx = shift_image(image, -1, 0), shift_image(image, 1, 0)
        Ipy, Imy = shift_image(image, -1, 1), shift_image(image, 1, 1)
        Ipz, Imz = shift_image(image, -1, 2), shift_image(image, 1, 2)

        Ipxpy, Ipxmy = shift_image(Ipx, -1, 1), shift_image(Ipx, 1, 1)
        Imxpy, Imxmy = shift_image(Imx, -1, 1), shift_image(Imx, 1, 1)
        Ipxpz, Ipxmz = shift_image(Ipx, -1, 2), shift_image(Ipx, 1, 2)
        Imxpz, Imxmz = shift_image(Imx, -1, 2), shift_image(Imx, 1, 2)
        Ipypz, Ipymz = shift_image(Ipy, -1, 2), shift_image(Ipy, 1, 2)
        Imypz, Imymz = shift_image(Imy, -1, 2), shift_image(Imy, 1, 2)

        Ixx = Ipx - 2 * image + Imx
        Iyy = Ipy - 2 * image + Imy
        Izz = Ipz - 2 * image + Imz

        Ixy = (Ipxpy - Ipxmy - Imxpy + Imxmy) / 4
        Ixz = (Ipxpz - Ipxmz - Imxpz + Imxmz) / 4
        Iyz = (Ipypz - Ipymz - Imypz + Imymz) / 4

        # Ixx, Ixy, Ixz = np.gradient(Ix)
        # _, Iyy, Iyz = np.gradient(Iy)
        # Izz = np.gradient(Iz, axis=2)

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
