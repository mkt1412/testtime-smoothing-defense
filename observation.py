import numpy as np
import torch
from torchvision import models
from cv2.ximgproc import anisotropicDiffusion
from util import load_image, get_paths_by_ext, normalize_image_to_tensor
from matplotlib import pyplot as plt
from scipy.special import softmax
from scipy.stats import entropy

DATA_DIR = '/media/fantasie/backup/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002-20/'
# DATA_DIR = '/media/fantasie/backup/data/ILSVRC2012/val_correct/'

if __name__ == "__main__":
	resnet152 = models.resnet152(pretrained=True).cuda().eval()
	for p in resnet152.parameters():
		p.requires_grad = False

	image_paths = get_paths_by_ext(DATA_DIR, ['JPEG', 'pkl'])
	image_path = image_paths[0]

	# Load input images
	if 'resnet152' in image_path:  # adversarial images, already resized
		image = load_image(image_path, resize=False)
	else:  # clean images, need resizing
		image = load_image(image_path, resize=True)

	iternum = 100
	plt.ion()
	for i in range(iternum):
		image = anisotropicDiffusion(np.transpose((image * 255).astype(np.uint8), (1, 2, 0)), alpha=0.1, K=20, niters=1) \
				/ 255.0
		plt.cla()
		plt.imshow(image)
		plt.title("Iteration %d" % i)

		image = np.transpose(image, (2, 0, 1)).astype(np.float32)

		# Normalize and convert to tensor
		output = resnet152(torch.unsqueeze(normalize_image_to_tensor(np.copy(image)), 0).cuda())
		probs = softmax(output.cpu().numpy()).squeeze()
		# topk = torch.topk(output, k=5)

		textstr = '\n'.join(("Entropy: %f" % (entropy(probs)),
							 "Class: %d, Confidence: %f" % (np.argmax(probs), np.max(probs))
							 ))

		print(textstr)
		plt.text(240, 0, textstr)
		plt.pause(0.5)

	plt.ioff()
	plt.show()
