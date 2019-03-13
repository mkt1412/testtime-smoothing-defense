import numpy as np
import torchvision.transforms as transforms
from PIL import Image
from matplotlib import pyplot as plt
import os
from glob import glob
import foolbox
from art.classifiers import PyTorchClassifier
from art.attacks.projected_gradient_descent import ProjectedGradientDescent
from art.attacks.fast_gradient import FastGradientMethod
from art.attacks.deepfool import DeepFool
from art.attacks.saliency_map import SaliencyMapMethod
import torch
from torchvision import models
import cv2
import pickle
import getpass

# Set up the root directory of ImageNet
# All directories must end with '/'
if getpass.getuser() == 'fantasie':  # user is Yifei
    ROOT_DIR = '/media/fantasie/backup/data/ILSVRC2012/'  # root directory for ImageNet
    SOURCE_DIR = ROOT_DIR + 'val/'  # directory for validation set
    CORRECT_DIR = ROOT_DIR + 'val_correct/'  # directory for correctly classified validation samples
    ADV_DIR = ROOT_DIR + 'val_correct_adv_resnet152_fast-gradient/'  # directory for precomputed adversarial examples
    NIPS17_DIR = '/media/fantasie/WD Elements/data/nips-2017-adversarial-learning-development-set/'  # NIPS-17 dataset
else:  # user is Chao
    ROOT_DIR = '/home/chaotang/PycharmProjects/data/ILSVRC2012/'
    SOURCE_DIR = ROOT_DIR + 'val/'
    CORRECT_DIR = ROOT_DIR + 'val_correct/'
    ADV_DIR = ROOT_DIR + 'val_correct_adv_resnet152_fast-gradient/'
    NIPS17_DIR = '/home/chaotang/PycharmProjects/data/NIPS2017/nips-2017-adversarial-learning-development-set/'


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


def load_image(img_path, resize=True, normalize=False):
    """
    Load image from a specified file path
    :param img_path: the file path of the image
    :param resize: whether or not to perform resize and centercrop, either do it here or with the model
    :param normalize: whether or not to perform normalization, either do it here or with the model
    :return: float32 numpy array with shape (3 x h x w)
    """
    img = Image.open(img_path)
    if img.mode != 'RGB':  # convert other modes with different number of channels to 3-channel'RGB'
        img = img.convert(mode='RGB')

    if resize:
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
        ])
        img = preprocess(img)

    if normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # converts to channel-first tensor
            normalize,
        ])
        img = np.array(preprocess(img))
    else:
        img = np.transpose(np.array(img), (2, 0, 1)).astype(np.float32) / 255.0  # channel first and [0, 1]

    return img


def load_pkl_image(img_path, normalize=False):
    """
    Load float images from a .pkl file that has been saved via save_array_to_pkl()
    Note that the image in .pkl has been already resized, there is no resize option available
    :param img_path: file path of the .pkl image file
    :param normalize: whether or not to perform normalization for Pytorch models, default False
    :return: numpy array with size (3 x h x w)
    """
    with open(img_path, "rb") as f:
        img = pickle.load(f)

    if normalize:
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        preprocess = transforms.Compose([
            transforms.ToTensor(),  # converts to channel-first tensor
            normalize,
        ])
        img = np.array(preprocess(img))

    return img


def load_image_batch(img_paths):
    """
    Load a batch of images
    :param img_paths: file paths of the images in the batch
    :return: numpy array with size (N x 3 x h x w)
    """
    imgs = []
    for img_path in img_paths:
        imgs.append(load_image(img_path))

    return np.stack(imgs, axis=0)


def numpy_to_tensor(arr):
    """
    Convert numpy-array image to a Pytorch tensor
    :param arr: numpy array representing an image, size (3 x h x w)
    :return: Pytorch tensor with size (None x 3 x h x w)
    """
    if len(arr.shape) < 4:
        arr = np.expand_dims(arr, axis=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.from_numpy(arr).to(device)


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


def load_nips17_labels(nips_dir=NIPS17_DIR):
    """
    Load the ground truth label and target class for NIPS17 DEV dataset
    :param nips_dir: directory of the NIPS17 DEV dataset
    :return: dictionary {key: image name w/o extension; value: [ground-truth label, target class])
    """
    import csv
    labels = {}
    with open(nips_dir + 'images.csv', newline='') as f:
        reader = csv.reader(f)
        _ = next(reader)  # skip the first row
        for row in reader:
            imgname, label, target = row[0], int(row[6]), int(row[7])
            labels[imgname] = [label, target]
    return labels


def subset_samping(list_pool, sample_num):
    '''
    Select random subset of Imagenet
    :param list_pool: paths of all the images
    :param sample_num: size of subset
    :return:
    '''
    import random
    random.seed(9012)
    random.shuffle(list_pool)

    return list_pool[0:sample_num]


def select_correct_images(source_dir=SOURCE_DIR, target_dir=CORRECT_DIR, save_files=False):
    """
    Scripts used for selecting samples that are correctly classified by a pretrained network
    :param source_dir: source directory that contains the universe of test images
    :param target_dir: target directory that used for saving correctly classified images when save_files is True
    :param save_files: indicator: whether or not to copy and store the correctly classified images
    :return: None
    """
    from shutil import copy2
    # Configure the pretrained model
    model = models.resnet152(pretrained=True).cuda().eval()

    image_paths = sorted(glob(os.path.join(source_dir + '**/*.JPEG'), recursive=True))

    i, count = 1, 0
    class_index_dict = map_class_indices()  # {n01440764: 0}
    for image_path in image_paths:
        print("i = %d" % i)
        i += 1

        image = load_image(image_path, True, True)
        prediction = torch.argmax(model(numpy_to_tensor(image)))

        code = os.path.basename(os.path.dirname(image_path))
        label = class_index_dict[code]

        if label == prediction:
            count += 1
            print("count = %d" % count)

            if save_files:
                parts = image_path.split('/')
                output_dir = os.path.join(target_dir, parts[-2])
                if not os.path.exists(output_dir):
                    os.makedirs(output_dir)

                copy2(image_path, output_dir)

    print("Test accuracy: ", count / len(image_paths))


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


def save_array_to_pkl(arr, fp):
    """
    Save a float array to the disk input arr is "passed by value"
    :param arr: numpy array, size (h x w x 3) or (3 x h x w)
    :param fp: target file path
    :return: None
    """
    directory = os.path.dirname(fp)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(fp, "wb") as f:
        pickle.dump(arr, f)


def save_adversarial_examples(clean_dir):
    """
    Scripts to compute and save adversarial examples with foolbox
    Need to manually set the models, directories and hyper-parameters
    Save "non-exisiting" adversarial examples without quantization in a .pkl file
    :param clean_dir: root directory of the legitimate clean data
    :return: None
    """
    dataset = 'ImageNet' if 'ILSVRC' in clean_dir else 'NIPS17'

    adv_dir = os.path.dirname(clean_dir) + "_adv_5000_resnet152_DeepFool"  # named after hyper-parameters

    # Load and create an instance of pretrained model
    resnet = models.resnet152(pretrained=True).cuda().eval()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model = foolbox.models.PyTorchModel(resnet, bounds=(0.0, 1.0), num_classes=1000, preprocessing=(mean, std))

    if dataset == 'ImageNet':  # ImageNet validation set
        image_paths = sorted(glob(clean_dir + '/**/*.JPEG', recursive=True))
        class_index_dict = map_class_indices()  # {n01440764: 0}
    else:  # NIPS 17 adversarial learning development set
        image_paths = sorted(glob(clean_dir + '/**/*.png', recursive=True))
        labels_dict = load_nips17_labels()

    image_paths = subset_samping(image_paths, sample_num=5050)

    count = 0
    for image_path in image_paths:
        # Get source image and label
        image = load_image(image_path)

        if dataset == 'ImageNet':
            code = os.path.basename(os.path.dirname(image_path))
            label = class_index_dict[code]
        else:
            label = labels_dict[os.path.splitext(os.path.basename(image_path))[0]][0]  # fn (w/o ext): [label, target]

        # Apply attack on source image
        attack = foolbox.attacks.DeepFoolAttack(model)
        adv_image = attack(image, label)

        output_path = os.path.join(adv_dir, os.path.splitext(image_path)[0][len(clean_dir):] + '.pkl')
        if adv_image is not None:  # sometimes adversarial attack return None
            save_array_to_pkl(adv_image, output_path)

        count += 1
        print(count)


def save_adversarial_examples_batch(clean_dir):
    """
    Scripts for pre-computing and saving adversarial examples with IBM-ART
    :param clean_dir: root directory of the legitimate clean data
    :return: None
    """
    batch_size = 16
    adv_dir = os.path.dirname(clean_dir) + "_adv_resnet152_SaliencyMap"  # named after hyper-parameters

    # Load pretrained model
    model = models.resnet152(pretrained=True).cuda().eval()
    # Build crafter for adversarial examples
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    classifier = PyTorchClassifier((0.0, 1.0), model=model, preprocessing=(mean, std),
                                   loss=torch.nn.modules.loss.CrossEntropyLoss(),
                                   optimizer=torch.optim.Adam,  # doesn't really matter for pretrained networks
                                   input_shape=(3, 224, 224), nb_classes=1000)
    adv_crafter = ProjectedGradientDescent(classifier)

    image_paths = sorted(glob(clean_dir + '/**/*.JPEG', recursive=True))
    count = 0
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        image_batch_paths = image_paths[start:end]
        images = load_image_batch(image_batch_paths)
        adv_images = adv_crafter.generate(x=images)

        assert len(adv_images) == len(images)

        print("start: %d, end: %d" % (start, end))
        for (image_path, adv_image) in zip(image_batch_paths, adv_images):
            output_path = os.path.join(adv_dir, image_path[len(clean_dir):])
            save_array_to_image(adv_image, output_path)

            count += 1
            print(count)


if __name__ == "__main__":
    save_adversarial_examples(clean_dir=CORRECT_DIR)
