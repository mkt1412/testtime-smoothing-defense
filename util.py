import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import os
from glob import glob
import foolbox
from torchvision import models
import cv2
import getpass

# Set up the root directory of ImageNet
if getpass.getuser() == 'fantasie':  # user is Yifei
    ROOT_DIR = '/media/fantasie/backup/data/ILSVRC2012/val_correct'
else:  # user is Chao
    ROOT_DIR = ''


def display_array_as_image(arr):
    """
    Display array as image, the input is "passed by value"
    :param arr: numpy array with range [0, 1] or [0, 255]
    :return: None
    """
    if arr.shape[0] < 4:  # channel first --> channel last
        arr = np.transpose(arr, (1, 2, 0))
    if np.max(arr) < 2.0:
        arr = arr * 255.0
    plt.imshow(arr.astype(np.uint8))


def load_image(img_path):
    """
    Load image from a specified file path
    :param img_path: the file path of the image
    :return: numpy array with shape (h x w x 3)
    """
    # Standard pytorch preprocessing
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ])

    img = Image.open(img_path)
    if img.mode != 'RGB':  # convert other modes with different number of channels to 3-channel'RGB'
        img = img.convert(mode='RGB')
    img = preprocess(img)
    img = np.transpose(np.array(img), (2, 0, 1)) / 255.0  # channel first and [0, 1]

    return img.astype(np.float32)


def map_class_indices():
    """
    Map the folder names (codes) to class indices
    :return: dictionary {key: folder name; value: index}
    """
    with open('synset_words.txt') as f:
        lines = f.readlines()
    class_index = {}
    count = 0
    for line in lines:
        parts = line.split()
        code = parts[0]
        class_index[code] = count
        count += 1

    return class_index


def save_array_to_image(arr, fp):
    """
    Save an numpy array as an image to the disk, input arr is "passed by value"
    :param arr: numpy array, size (h x w x 3) or (3 x h x w)
    :param fp: target file path
    :return: None
    """
    directory = os.path.dirname(fp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    if arr.shape[0] < 4:  # channel first --> channel last
        arr = np.transpose(arr, (1, 2, 0))

    arr = (arr * 255).astype(np.uint8)[:, :, ::-1]
    cv2.imwrite(fp, arr)


def save_adversarial_examples(source_dir):
    """
    Scripts to compute and save adversarial examples. Need to manually set the models, directories and hyper-parameters
    :param source_dir: root directory of the legitimate clean data
    :return: None
    """
    root_dir = os.path.dirname(source_dir)
    sub_dir = os.path.basename(source_dir) + "_adv_resnet152_gradientsign"  # named after hyper-parameters
    target_dir = os.path.join(root_dir, sub_dir)

    # Load and create an instance of pretrained model
    resnet152 = models.resnet152(pretrained=True).cuda().eval()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model = foolbox.models.PyTorchModel(resnet152, bounds=(0.0, 1.0), num_classes=1000, preprocessing=(mean, std))

    class_index_dict = map_class_indices()  # {n01440764: 0}

    image_paths = sorted(glob(source_dir + '/**/*.JPEG', recursive=True))
    count = 0
    for image_path in image_paths:
        # Get source image and label
        image = load_image(image_path)
        code = os.path.basename(os.path.dirname(image_path))
        label = class_index_dict[code]

        # Apply attack on source image
        attack = foolbox.attacks.GradientSignAttack(model)
        adv_image = attack(image, label)  # max_epsilon=0.5, epsilons=1)

        output_path = os.path.join(target_dir, image_path[len(source_dir) + 1:])
        if adv_image is not None:  # sometimes adversarial attack return None
            save_array_to_image(adv_image, output_path)

        count += 1
        print(count)


if __name__ == "__main__":
    save_adversarial_examples(ROOT_DIR)
