from glob import glob
from torchvision import models
import foolbox
import numpy as np
import os
from medpy.filter import anisotropic_diffusion
from util import display_array_as_image, load_image, load_pkl_image, map_class_indices, load_nips17_labels
from skimage.restoration import denoise_nl_means, denoise_bilateral
import time
from scipy.ndimage.filters import median_filter, convolve, gaussian_filter
from modified_curvature_motion import modified_curvature_motion
import getpass
import argparse

# Configuration on directories
if getpass.getuser() == 'fantasie':  # user is Yifei
    DATA_DIR = '/media/fantasie/backup/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002/'
else:  # user is Chao
    DATA_DIR = '/home/chaotang/PycharmProjects/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002/'

# Parsing input arguments
parser = argparse.ArgumentParser(description='Smoothing against adversarial examples')
parser.add_argument('--dir', '--data-dir', default=DATA_DIR, type=str,
                    metavar='DIR', help='path to dataset', dest='data_dir')
parser.add_argument('--d', '--defense', default=None, type=str,
                    metavar='DEFENSE', help='defense method', dest='defense')
parser.add_argument('--p', '--param', default=None, type=float, nargs='+',
                    metavar='PARAMETER', help='defense parameters', dest='param')  # for finding optimal hyper-params


if __name__ == "__main__":
    # Parsing input arguments
    args = parser.parse_args()
    print("*************************************")
    print("Target directory: %s" % args.data_dir)
    print("Defense method: ", args.defense, ", with parameters ", args.param)

    # Configure dataset
    DATASET = "ImageNet"
    if DATASET == "ImageNet":
        class_index_dict = map_class_indices()
    else:
        labels_dict = load_nips17_labels()

    # Load pretrained model
    resnet152 = models.resnet152(pretrained=True).cuda().eval()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model = foolbox.models.PyTorchModel(resnet152, bounds=(0.0, 1.0), num_classes=1000, preprocessing=(mean, std))

    # File paths of the target images
    image_paths = sorted(glob(args.data_dir + '**/*.JPEG', recursive=True))

    i, count = 1, 0
    start_time = time.time()

    for image_path in image_paths:
        # print("i = %d" % i)
        i += 1

        # Load input images
        if image_path.endswith('pkl'):
            image = load_pkl_image(image_path)
        else:
            if 'resnet152' in image_path:  # adversarial images, already resized
                image = load_image(image_path, resize=False)
            else:  # clean images, need resizing
                image = load_image(image_path, resize=True)

        # Choose smoothing methods and apply
        if args.defense == 'anisotropic_diffusion':
            image = anisotropic_diffusion(image, niter=10).astype("float32")
        elif args.defense == 'modified_curvature_motion':
            image = modified_curvature_motion(image, k=args.param[0], niter=int(args.param[1]))
        elif args.defense == 'mean':
            image = convolve(image, weights=[1, 3, 3, 3] / 27.0)
        elif args.defense == 'median':
            image = median_filter(image)
        elif args.defense == 'non_local_mean':
            image = np.transpose(denoise_nl_means(np.transpose(image, (1, 2, 0)), multichannel=True),
                                 (2, 1, 0)).astype("float32")
        elif args.defense == 'bilateral':
            image = denoise_bilateral(image.astype(np.double), multichannel=True).astype(np.float32)
        elif args.defense == 'gaussian':
            image = gaussian_filter(image, 3).astype("float32")
        else:
            pass

        # Get the output class
        prediction = np.argmax(model.predictions(image))

        if DATASET == "ImageNet":
            code = os.path.basename(os.path.dirname(image_path))
            label = class_index_dict[code]
        else:
            label = labels_dict[os.path.splitext(os.path.basename(image_path))[0]][0]  # fn (w/o ext): [label, target]

        if label == prediction:
            count += 1
            # print("count = %d" % count)
        # print("percentage = %f" % (count / i))

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time:", total_time)
    print("Test accuracy: ", count/len(image_paths))
