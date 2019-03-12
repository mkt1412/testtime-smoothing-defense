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
import torch
from torchvision import models
import cv2
import getpass

# Set up the root directory of ImageNet
if getpass.getuser() == 'fantasie':  # user is Yifei
    ROOT_DIR = '/media/fantasie/backup/data/ILSVRC2012/'  # root directory for ImageNet
    SOURCE_DIR = ROOT_DIR + 'val/'  # directory for validation set
    CORRECT_DIR = ROOT_DIR + 'val_correct1/'  # directory for correctly classified validation samples
    ADV_DIR = ROOT_DIR + 'val_correct_adv_resnet152_fast-gradient/'  # directory for precomputed adversarial examples
else:  # user is Chao
    ROOT_DIR = ''
    SOURCE_DIR = ROOT_DIR + 'val'
    CORRECT_DIR = ROOT_DIR + 'val_correct'
    ADV_DIR = ROOT_DIR + ''


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


def save_adversarial_examples(source_dir):
    """
    Scripts to compute and save adversarial examples with foolbox
    Need to manually set the models, directories and hyper-parameters
    Note:
    Discarded because the execution time is too long for sophisticated attacks such as PGD, and no clipping is provided
    to account for quantization
    :param source_dir: root directory of the legitimate clean data
    :return: None
    """
    root_dir = os.path.dirname(source_dir)
    sub_dir = os.path.basename(source_dir) + "_adv_resnet152_test"  # named after hyper-parameters
    target_dir = os.path.join(root_dir, sub_dir)

    # Load and create an instance of pretrained model
    resnet = models.resnet152(pretrained=True).cuda().eval()
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    model = foolbox.models.PyTorchModel(resnet, bounds=(0.0, 1.0), num_classes=1000, preprocessing=(mean, std))

    class_index_dict = map_class_indices()  # {n01440764: 0}

    image_paths = sorted(glob(source_dir + '/**/*.JPEG', recursive=True))
    count = 0
    for image_path in image_paths[0:50]:
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


def save_adversarial_examples_batch(source_dir):
    batch_size = 16
    root_dir = os.path.dirname(source_dir)
    sub_dir = os.path.basename(source_dir) + "_adv_resnet152_pgd"  # named after hyper-parameters
    target_dir = os.path.join(root_dir, sub_dir)

    # Load pretrained model
    model = models.resnet152(pretrained=True).cuda().eval()
    # Build crafter for adversarial examples
    mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
    std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
    classifier = PyTorchClassifier((0.0, 1.0), model=model, preprocessing=(mean, std),
                                   loss=torch.nn.modules.loss.CrossEntropyLoss(),
                                   optimizer=torch.optim.Adam,
                                   input_shape=(3, 224, 224), nb_classes=1000)
    adv_crafter = ProjectedGradientDescent(classifier)

    image_paths = sorted(glob(ROOT_DIR + '/**/*.JPEG', recursive=True))
    count = 0
    for start in range(0, len(image_paths), batch_size):
        end = min(start + batch_size, len(image_paths))
        image_batch_paths = image_paths[start:end]
        images = load_image_batch(image_batch_paths)
        adv_images = adv_crafter.generate(x=images)

        assert len(adv_images) == len(images)

        print("start: %d, end: %d" % (start, end))
        for (image_path, adv_image) in zip(image_batch_paths, adv_images):
            output_path = os.path.join(target_dir, image_path[len(source_dir) + 1:])
            save_array_to_image(adv_image, output_path)

            count += 1
            print(count)


if __name__ == "__main__":
    save_adversarial_examples_batch(ROOT_DIR)
