from glob import glob
from torchvision import models
import foolbox
import numpy as np
import os
from scipy.signal import medfilt
from medpy import filter as filter
from util import display_array_as_image, load_image, load_pkl_image, map_class_indices, load_nips17_labels
import time
from scipy.ndimage.filters import median_filter, convolve


# Attacked_Dir := root folder of adversarial examples
# img_paths = glob(path_to_val)
# For each image img in /IL...2012/val (for img_path in img_paths)
#     img_adv = fgsm(img)
#     save(img_adv, path=os.path.join(Attacked_Dir, img_path))

# Configure dataset
DATASET = "ImageNet"
if DATASET == "ImageNet":
    class_index_dict = map_class_indices()
else:
    labels_dict = load_nips17_labels()


resnet152 = models.resnet152(pretrained=True).cuda().eval()
mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))
model = foolbox.models.PyTorchModel(resnet152, bounds=(0.0, 1.0), num_classes=1000, preprocessing=(mean, std))


image_paths = sorted(glob(r'/home/chao/PycharmProjects/data/ILSVRC2012'
                          r'/val_correct_adv_resnet152_pgd-0.01-0.002/**/*.JPEG',
                          recursive=True))

i, count = 1, 0
start_time = time.time()

for image_path in image_paths:
    print("i = %d" % i)
    i += 1

    if image_path.endswith('pkl'):
        image = load_pkl_image(image_path)
    else:
        if 'resnet152' in image_path:  # adversarial images, already resized
            image = load_image(image_path, resize=False)
        else:  # clean images, need resizing
            image = load_image(image_path, resize=True)
    # image = filter.anisotropic_diffusion(image, niter=2, option=1).astype("float32")
    # image = median_filter(image, size=(1, 3, 3, 3))

    prediction = np.argmax(model.predictions(image))

    if DATASET == "ImageNet":
        code = os.path.basename(os.path.dirname(image_path))
        label = class_index_dict[code]

    else:
        label = labels_dict[os.path.splitext(os.path.basename(image_path))[0]][0]  # fn (w/o ext): [label, target]

    if label == prediction:
        count += 1
        print("count = %d" % count)

end_time = time.time()
total_time = end_time - start_time
print("Total time:", total_time)
print("Test accuracy: ", count/len(image_paths))
