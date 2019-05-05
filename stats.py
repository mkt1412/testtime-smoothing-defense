"""
record the maximum iteration number to achieve correct prediction using
"""
import numpy as np
from cv2.ximgproc import anisotropicDiffusion
from torchvision import models
import getpass
from util import load_image, get_paths_by_ext, normalize_image_to_tensor, map_class_indices, load_nips17_labels, \
    save_array_to_pkl
from scipy.special import softmax
import torch
import os
from anisotropic_diffusion import anisotropic_diffusion_3d
from scipy.ndimage.filters import convolve


if getpass.getuser() == 'fantasie':  # user is Yifei
    DATA_DIR = '/media/fantasie/backup/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002-20/'
    RESULT_DIR = 'result/'
else:  # user is Chao
    DATA_DIR = '/home/chao/PycharmProjects/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002-20/'
    RESULT_DIR = '/home/chao/PycharmProjects/data/ILSVRC2012/result/'


if __name__ == "__main__":
    resnet152 = models.resnet152(pretrained=True).cuda().eval()
    for p in resnet152.parameters():
        p.requires_grad = False

    image_paths = get_paths_by_ext(DATA_DIR, ['JPEG', 'pkl'])

    # Configure dataset
    DATASET = "ImageNet"
    if DATASET == "ImageNet":
        class_index_dict = map_class_indices()
    else:
        labels_dict = load_nips17_labels()

    maxiter = 10
    iter_dist = np.full(shape=len(image_paths), fill_value=998)
    count = 0
    for image_path in image_paths:
        print(count)
        # Load input images
        if 'resnet152' in image_path:  # adversarial images, already resized
            ori_image = (load_image(image_path, resize=False) * 255).astype(np.uint8) / 255.0
        else:  # clean images, need resizing
            ori_image = load_image(image_path, resize=True)

        # Load true label
        if DATASET == "ImageNet":
            code = os.path.basename(os.path.dirname(image_path))
            label = class_index_dict[code]
        else:
            label = labels_dict[os.path.splitext(os.path.basename(image_path))[0]][0]

        for i in range(1, maxiter+1):
            # image = anisotropicDiffusion(np.transpose((ori_image * 255).astype(np.uint8), (1, 2, 0)), alpha=0.1, K=20,
            #                              niters=i) / 255.0
            # image = np.transpose(image, (2, 0, 1)).astype(np.float32)
            # image = anisotropic_diffusion_3d(input=image, gamma=0.1, kappa=20, niter=1).astype(np.float32)
            image = convolve(ori_image, weights=np.full((1, i, i), 1.0 / i ** 2)).astype(np.float32)

            # Make prediction
            output = resnet152(torch.unsqueeze(normalize_image_to_tensor(np.copy(image)), 0).cuda())
            probs = softmax(output.cpu().numpy()).squeeze()
            prediction = np.argmax(probs)

            # Compare prediction to true label
            if label == prediction:
                iter_dist[count] = i
                break

        count += 1

    result_fp = RESULT_DIR + DATA_DIR.split('/')[-2] + '_' + "miniter" + '_' + "mean" + '.pkl'
    save_array_to_pkl(iter_dist, result_fp)














