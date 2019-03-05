from glob import glob
from torchvision import models
import foolbox
import torch
import torch.nn as nn
import numpy as np
import os
import cv2
from shutil import copy2
import torchvision.transforms as transforms
from PIL import Image

# Attacked_Dir := root folder of adversarial examples
# img_paths = glob(path_to_val)
# For each image img in /IL...2012/val (for img_path in img_paths)
#     img_adv = fgsm(img)
#     save(img_adv, path=os.path.join(Attacked_Dir, img_path))


def load_image(img_path):
    """
    Load image from a specified file path
    :param img_path: the file path of the image
    :return: numpy array with shape (h x w x 3)
    """
    # Standard pytorch preprocessing as tensor
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize
    ])

    img = Image.open(img_path)
    if img.mode != 'RGB':  # convert other modes with different number of channels to 3-channel'RGB'
        img = img.convert(mode='RGB')
    img = preprocess(img)
    img = img.numpy()  # convert back to numpy array

    return img


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


class_index_dict = map_class_indices()

resnet152 = models.resnet152(pretrained=True).cuda().eval()
model = foolbox.models.PyTorchModel(resnet152, bounds=(0.0, 1.0), num_classes=1000, preprocessing=(0, 1))

attack = foolbox.attacks.GradientSignAttack(model)
attack_dir = "./attack_dir/"

image_paths = glob(r'/media/fantasie/backup/data/ILSVRC2012/val/**/*.JPEG', recursive=True)
target_dir = '/media/fantasie/backup/data/ILSVRC2012/val_correct'

i, count = 1, 0

for image_path in image_paths:
    print("i = %d" % i); i += 1

    image = load_image(image_path)
    prediction = np.argmax(model.predictions(image))

    code = os.path.basename(os.path.dirname(image_path))
    label = class_index_dict[code]

    if label == prediction:
        count += 1
        print("count = %d" % count)

        parts = image_path.split('/')
        output_dir = os.path.join(target_dir, parts[-2])
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        copy2(image_path, output_dir)

    #adv_image = attack(image, label, max_epsilon=0.5, epsilons=1)
    # adv_image = image
    # class_dir = os.path.join(attack_dir, code)
    # if not(os.path.exists(class_dir)):
    #     os.mkdir(class_dir)
    # cv2.imwrite(filename=os.path.join(class_dir, os.path.basename(image_path)), img=np.uint8(np.transpose(adv_image, (2, 1, 0))*255))

print("Test accuracy: ", count/len(image_paths))
