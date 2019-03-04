from glob import glob
from torchvision import models
import foolbox
import torch
import torch.nn as nn
import numpy as np
import os
import cv2

# Attacked_Dir := root folder of adversarial examples
# img_paths = glob(path_to_val)
# For each image img in /IL...2012/val (for img_path in img_paths)
#     img_adv = fgsm(img)
#     save(img_adv, path=os.path.join(Attacked_Dir, img_path))


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

resnet18 = models.resnet18(pretrained=True).cuda().eval()
mean = np.array([0.485, 0.456, 0.406]).reshape((3,1,1))
std = np.array([0.229, 0.224, 0.225]).reshape((3,1,1))
model = foolbox.models.PyTorchModel(resnet18, bounds=(0,1), num_classes=1000, preprocessing=(mean, std))

attack = foolbox.attacks.GradientSignAttack(model)

attack_dir = "./attack_dir/"
image_paths = glob(r'/home/chaotang/Desktop/special_problem/test/**/*.JPEG', recursive=True)

for image_path in image_paths:
    image = cv2.imread(image_path)
    image = (np.transpose(image, (2, 1, 0))/255).astype("float32")
    image = np.resize(image, (3,224, 224))
    code = os.path.basename(os.path.dirname(image_path))
    label = class_index_dict[code]
    adv_image = attack(image, label, max_epsilon=0.5, epsilons=1)
    class_dir = os.path.join(attack_dir, code)
    if not(os.path.exists(class_dir)):
        os.mkdir(class_dir)
    cv2.imwrite(filename=os.path.join(class_dir, os.path.basename(image_path)), img=np.uint8(adv_image*255))
