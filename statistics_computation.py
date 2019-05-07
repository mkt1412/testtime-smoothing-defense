import pickle
import numpy as np
from matplotlib import pyplot as plt
from util import get_paths_by_ext, map_class_indices, save_array_to_pkl,  load_image, normalize_image_to_tensor
import os
from torchvision import models
import torch
from scipy.special import softmax


TEST_DIR = "/home/chao/PycharmProjects/data/ILSVRC2012/val_correct/"
RESULT_DIR = "/home/chao/PycharmProjects/data/ILSVRC2012/result/"
DATA_DIR = '/home/chao/PycharmProjects/data/ILSVRC2012/val_correct_adv_resnet152_pgd-0.01-0.002-20/'


def ground_truth_prob():
    """
    Compute ground truth prob for adversarial examples
    :return: None
    """
    resnet152 = models.resnet152(pretrained=True).cuda().eval()
    for p in resnet152.parameters():
        p.requires_grad = False

    image_paths = get_paths_by_ext(DATA_DIR, ['JPEG', 'pkl'])
    class_index_dict = map_class_indices()

    ground_truth_prob_list = np.zeros(len(image_paths))

    count = 0

    for image_path in image_paths:
        print(count)
        if 'resnet152' in image_path:  # adversarial images, already resized
            image = load_image(image_path, resize=False)
        else:  # clean images, need resizing
            image = load_image(image_path, resize=True)

        # Map the ground truth label to index
        code = os.path.basename(os.path.dirname(image_path))
        label = class_index_dict[code]

        # Normalize and convert to tensor
        output = resnet152(torch.unsqueeze(normalize_image_to_tensor(np.copy(image)), 0).cuda())
        probs = softmax(output.cpu().numpy()).squeeze()
        ground_truth_prob_list[count] = probs[label]

        result_fp = RESULT_DIR + 'pgd-0.01-0.002' + '_' + 'ground_truth' + '_' + 'prob' + '.pkl'
        save_array_to_pkl(ground_truth_prob_list, result_fp)

        count += 1


def show_entropy(img_path, num=10):
    """
    Show the trend of entropy as number of iterations getting larger (1-20)
    :param: img_path: path to entropy file
    :param num: number of samples to show
    :return: None
    """
    img_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/entropy/val_correct_adv_resnet152_pgd-0.05-0.01.pkl"
    with open(img_path, "rb") as f:
        entropy_list = pickle.load(f)
    plt.figure("entropy")
    x = np.linspace(start=1, stop=20, num=20)
    for i in range(num):
        y = np.array(entropy_list)[i]
        plt.plot(x, y)
    plt.show()

    return None


def summarize_categorical_results(result_path, test_dir=TEST_DIR):
    """
    Compute summary statistics for each category
    :param result_path: file path of the .pkl file that stores the prediction confidence(+)/illusiveness(-)
    :param test_dir: root directory of the test images
    :return: numpy array of categorical accuracy (1000, )
    """
    with open(result_path, "rb") as f:
        results = pickle.load(f)

    img_paths = get_paths_by_ext(test_dir, ['JPEG', 'pkl'])
    results_categorical, amount = np.zeros(1000), np.zeros(1000)
    class_index_dict = map_class_indices()

    for result, img_path in zip(results, img_paths):
        code = os.path.basename(os.path.dirname(img_path))
        idx = class_index_dict[code]

        amount[idx] += 1
        results_categorical[idx] += result

    results_categorical /= amount

    return results_categorical


def acc_vs_confidence(acc_path, confidence_path):
    """
    Show the relationship between acc of adversarial categories and confidence of original categories
    :param acc_path: path to accuracy of adversarial examples
    :param confidence_path: path to confidence of original examples
    :return: None
    """
    with open(acc_path, "rb") as acc:
        acc_list = pickle.load(acc)
    with open(confidence_path, "rb") as confidences:
        confidence_list = pickle.load(confidences)

    # category-accuracy-confidence
    keys = np.linspace(start=1, stop=1000, num=1000)
    combo = np.vstack((np.vstack((keys, acc_list)), confidence_list)).T
    combo_sorted = combo[combo[:, 1].argsort()]  # sort the combo according to prediction acc

    data1 = combo_sorted[:, 1]  # sorted acc
    data2 = combo_sorted[:, 2]  # sorted confidence
    bins = np.linspace(start=1, stop=1000, num=1000)

    plt.figure("accuracy")
    plt.title("accuracy")
    plt.bar(x=bins, height=data1, label="acc", color="orange")
    plt.xlabel('category')
    plt.ylabel('percentage')
    plt.legend(loc='lower left')
    plt.ylim([0, 1])
    plt.xlim([-10, 1010])
    plt.figure("ground truth prob")
    plt.title("ground truth prob")
    plt.bar(x=bins, height=data2, label="confidence")
    plt.xlabel('category')
    plt.ylabel('percentage')
    plt.legend(loc='lower left')
    plt.ylim([0, 0.1])
    plt.xlim([-10, 1010])
    plt.show()

    return None


def optimal_subset_selection(confidence_path):
    """
    Select optimal subset for each attack and defense
    :param confidence_path: path to confidence pkl file
    :return: None
    """
    with open(confidence_path, "rb") as f:
        confidence_list = pickle.load(f)

    i = 0
    optimal_list = []
    img_paths = get_paths_by_ext(TEST_DIR, ['JPEG', 'pkl'])

    for img_path, confidence in zip(img_paths, confidence_list):
        if confidence > 0:
            optimal_list.append(i)
        i += 1

    return optimal_list


def optimal_subset_test(optimal_index_path, confidence_path):
    """
    Test B and C's acc on A's optimal subset
    :param optimal_index_path: path to A's optimal subset index
    :param confidence_path: path to B and C's confidence path
    :return: None
    """
    with open(optimal_index_path, "rb") as f:
        optimal_indexes = pickle.load(f)
    with open(confidence_path, "rb") as g:
        confidences = pickle.load(g)

    count = 0

    for optimal_index in optimal_indexes:
        if confidences[optimal_index] > 0:
            count += 1
    acc = count / len(optimal_indexes)

    print("pgd-0.01-0.002 using mean optimal subset on diffusion acc: {}".format(acc))

    return None


if __name__ == "__main__":
    # img_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/entropy/" \
    #            "val_correct_adv_resnet152_pgd-0.01-0.002-20.pkl"
    # show_entropy(img_path=img_path, num=20)

    # acc_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/defend_acc_vs_ground_truth_prob/" \
    #            "anisotropic_diffusion/anisotropic-diffusion_pgd-0.05-0.01_categorical_acc.pkl"
    # confidence_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/defend_acc_vs_ground_truth_prob/" \
    #                   "pgd-0.05-0.01_categorical_ground_truth_prob.pkl"
    # acc_vs_confidence(acc_path=acc_path, confidence_path=confidence_path)

    # result_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/pgd-0.05-0.01_ground_truth_prob.pkl"
    # ground_truth_prob_categorical = summarize_categorical_results(result_path=result_path)
    # result_fp = RESULT_DIR + 'pgd-0.05-0.01' + '_' + 'categorical' + '_' + 'ground_truth' + '_' + 'prob' + '.pkl'
    # save_array_to_pkl(ground_truth_prob_categorical, result_fp)

    # confidence_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/optimal_subset/pgd-0.01-0.002/mean/" \
    #                   "val_correct_adv_resnet152_pgd-0.01-0.002-20_mean_[1.0, 1.0, 5.0, 5.0].pkl"
    # optimal_list = optimal_subset_selection(confidence_path=confidence_path)
    # result_fp = RESULT_DIR + 'mean' + '_' + 'pgd-0.01-0.002' + '_' + 'optimal_index' + '.pkl'
    # save_array_to_pkl(optimal_list, result_fp)

    confidence_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/optimal_subset/pgd-0.05-0.01/" \
                      "anisotropic-diffusion/" \
                      "val_correct_adv_resnet152_pgd-0.05-0.01_anisotropic-diffusion_[0.1, 20.0, 7.0].pkl"
    optimal_index_path = "/home/chao/PycharmProjects/data/ILSVRC2012/result/optimal_subset/pgd-0.05-0.01/mean/" \
                         "mean_pgd-0.05-0.01_optimal_index.pkl"
    optimal_subset_test(optimal_index_path=optimal_index_path, confidence_path=confidence_path)





