from glob import glob
from torchvision import models
import foolbox
import numpy as np
import os
from scipy.signal import medfilt
from medpy import filter as filter
from util import display_array_as_image, load_image, map_class_indices


# Attacked_Dir := root folder of adversarial examples
# img_paths = glob(path_to_val)
# For each image img in /IL...2012/val (for img_path in img_paths)
#     img_adv = fgsm(img)
#     save(img_adv, path=os.path.join(Attacked_Dir, img_path))


class_index_dict = map_class_indices()


resnet152 = models.resnet152(pretrained=True).cuda().eval()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
model = foolbox.models.PyTorchModel(resnet152, bounds=(0.0, 1.0), num_classes=1000, preprocessing=(mean, std))

image_paths = glob(r'/media/fantasie/backup/data/ILSVRC2012/val_correct/**/*.JPEG', recursive=True)

i, count = 1, 0

for image_path in image_paths:
    print("i = %d" % i)
    i += 1

    image = load_image(image_path)
    # image = medfilt(image).astype("float32")
    image = filter.anisotropic_diffusion(image).astype("float32")

    prediction = np.argmax(model.predictions(image))

    code = os.path.basename(os.path.dirname(image_path))
    label = class_index_dict[code]

    if label == prediction:
        count += 1
        print("count = %d" % count)

print("Test accuracy: ", count/len(image_paths))
