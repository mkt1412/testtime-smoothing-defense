from glob import glob
from torchvision import models
import foolbox
import numpy as np
import os
from cv2 import bilateralFilter, fastNlMeansDenoisingColored
from cv2.ximgproc import anisotropicDiffusion
from medpy.filter import anisotropic_diffusion
from util import load_image, load_pkl_image, map_class_indices, load_nips17_labels, save_array_to_pkl, \
    display_array_as_image, get_paths_by_ext
from skimage.restoration import denoise_nl_means, denoise_bilateral
import time
from scipy.ndimage.filters import median_filter, convolve, gaussian_filter
from modified_curvature_motion import modified_curvature_motion
from scipy.special import softmax
import getpass
import argparse

# Configuration on directories: must end with '/'
if getpass.getuser() == 'fantasie':  # user is Yifei
    DATA_DIR = '/media/fantasie/backup/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002/'
    RESULT_DIR = 'result/'
else:  # user is Chao
    DATA_DIR = '/home/chao/PycharmProjects/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002/'
    RESULT_DIR = None

# Parsing input arguments
parser = argparse.ArgumentParser(description='Smoothing against adversarial examples')
parser.add_argument('--dir', '--data-dir', default=DATA_DIR, type=str,
                    metavar='DIR', help='path to dataset', dest='data_dir')
parser.add_argument('--def', '--defense', default=None, type=str,
                    metavar='DEFENSE', help='defense method', dest='defense')
parser.add_argument('--p', '--param', default=None, type=float, nargs='+',
                    metavar='PARAMETER', help='defense parameters', dest='param')  # for finding optimal hyper-params
parser.add_argument('--live', dest='live', action='store_true',
                    help='display live result')
parser.add_argument('--save', dest='save', action='store_true',
                    help='save test result')


if __name__ == "__main__":
    # Parsing input arguments
    args = parser.parse_args()

    # %% For debugging: manual configuration
    # args.defense = 'bilateral'

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
    image_paths = get_paths_by_ext(args.data_dir, ['JPEG', 'pkl'])

    i, count = 0, 0
    confidences = np.zeros(len(image_paths))
    start_time = time.time()

    for image_path in image_paths:
        if args.live:
            print("i = %d" % i)

        # Load input images
        if image_path.endswith('pkl'):
            image = load_pkl_image(image_path)
        else:
            if 'resnet152' in image_path:  # adversarial images, already resized
                image = load_image(image_path, resize=False)
            else:  # clean images, need resizing
                image = load_image(image_path, resize=True)

        # Choose smoothing methods and apply
        if args.defense == 'anisotropic-diffusion':
            image = anisotropicDiffusion(np.transpose((image * 255).astype('uint8'), (1, 2, 0)),
                                         alpha=args.param[0], K=args.param[1], niters=int(args.param[2]))
            image = np.transpose(image, (2, 0, 1)).astype("float32") / 255
        elif args.defense == 'modified-curvature-motion':
            image = modified_curvature_motion(image, niter=int(args.param[0]), k=args.param[1])
        elif args.defense == 'mean':
            image = convolve(image, weights=[1, 3, 3, 3] / 27.0)
        elif args.defense == 'median':
            image = median_filter(image)
        elif args.defense == 'non-local-mean':  # very slow
            image = np.transpose(fastNlMeansDenoisingColored(np.transpose((image * 255).astype('uint8'), (1, 2, 0)),
                                                             3, 3, 7, 21),  # default
                                 (2, 1, 0)).astype("float32") / 255
        elif args.defense == 'bilateral':
            image = np.transpose(bilateralFilter(np.transpose(image, (1, 2, 0)), 5, 75, 75),
                                 (2, 0, 1))
        elif args.defense == 'gaussian':
            image = gaussian_filter(image, 3).astype("float32")
        else:
            pass

        # Get the output class and confidence
        output = model.predictions(image)
        prediction = np.argmax(output)
        confidence = softmax(output)[prediction]

        # Map the ground truth label to index
        if DATASET == "ImageNet":
            code = os.path.basename(os.path.dirname(image_path))
            label = class_index_dict[code]
        else:
            label = labels_dict[os.path.splitext(os.path.basename(image_path))[0]][0]  # fn (w/o ext): [label, target]

        if label == prediction:
            count += 1
            if args.live:
                print("count = %d" % count)
            confidences[i] = confidence
        else:
            confidences[i] = -confidence  # set the confidence ("illusiveness") to negative for misclassified samples

        i += 1
        if args.live:
            print("percentage = %f" % (count / i))

    end_time = time.time()
    total_time = end_time - start_time
    print("Total time:", total_time)
    print("Test accuracy: ", count/len(image_paths))

    # Save test results for each sample (in a sorted sequence)
    if args.save and RESULT_DIR is not None:
        result_fp = RESULT_DIR + DATA_DIR.split('/')[-2] + '_' + args.defense + '_' + str(args.param) + '.pkl'
        save_array_to_pkl(confidences, result_fp)
