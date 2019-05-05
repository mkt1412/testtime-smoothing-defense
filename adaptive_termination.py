import numpy as np
import torch
from torchvision import models
from util import get_paths_by_ext, load_image, normalize_image_to_tensor, save_data_to_pkl
from cv2.ximgproc import anisotropicDiffusion
from scipy.special import softmax
from scipy.stats import entropy
import os

ROOT_DIR = '/media/fantasie/backup/data/ILSVRC2012/'
DATA_DIR = ROOT_DIR + 'val_correct_adv_resnet152_pgd-0.05-0.01/'
RESULT_DIR = '/media/fantasie/backup/result/smoothing-against-adversarial-examples/records/'

if __name__ == "__main__":
    resnet152 = models.resnet152(pretrained=True).cuda().eval()
    for p in resnet152.parameters():
        p.requires_grad = False

    image_paths = get_paths_by_ext(DATA_DIR, ['JPEG', 'pkl'])

    entropy_record = []  # list of lists of entropy, for each image and for each iteration

    for i, image_path in enumerate(image_paths):
        print(i)
        prob_record = []  # list of probability vector in numpy arrays, for each image
        entropy_record.append([])

        # Load input images
        if 'resnet152' in image_path:  # adversarial images, already resized
            image = load_image(image_path, resize=False)
        else:  # clean images, need resizing
            image = load_image(image_path, resize=True)

        for niter in range(1, 21):
            image_ = anisotropicDiffusion(np.transpose((image * 255).astype(np.uint8), (1, 2, 0)),
                                          alpha=0.1, K=20, niters=niter) / 255.0
            image_ = np.transpose(image_, (2, 0, 1)).astype(np.float32)

            output = resnet152(torch.unsqueeze(normalize_image_to_tensor(np.copy(image_)), 0).cuda())
            probs = softmax(output.cpu().numpy()).squeeze()
            entr = entropy(probs)

            prob_record.append(probs)
            entropy_record[i].append(entr)

        # Save the probabilities from each iteration, for each image
        output_path = os.path.splitext(os.path.join(RESULT_DIR, image_path[len(ROOT_DIR):]))[0] + '.pkl'
        save_data_to_pkl(prob_record, output_path)

    # Save the entropy evolution for the entire dataset
    output_path = RESULT_DIR + DATA_DIR.split('/')[-2] + '.pkl'
    save_data_to_pkl(entropy_record, output_path)

    pass
