import os
from sklearn import config_context
import torch
import random
from torchvision.utils import save_image
import numpy as np
import config
from torchvision import transforms
from PIL import Image

"""Adaptive backdoor attack (with k triggers)
Just keep the original labels for some (say 50%) poisoned samples...

Poison with k triggers.
"""
# k = 4
# trigger_names = [
#     'hellokitty_split_1_32.png',
#     'hellokitty_split_2_32.png',
#     'hellokitty_split_3_32.png',
#     'hellokitty_split_4_32.png',
#     # 'hellokitty_r_32.png',
#     # 'hellokitty_g_32.png',
#     # 'hellokitty_b_32.png',
# ]
# alphas = [
#     0.2,
#     0.2,
#     0.2,
#     0.2,
#     # 0.2,
#     # 0.2,
#     # 0.2,
# ]

# test_k = 1
# test_trigger_names = [
#     # 'hellokitty_split_1_32.png',
#     # 'hellokitty_split_2_32.png',
#     # 'hellokitty_split_3_32.png',
#     # 'hellokitty_split_4_32.png',
#     'hellokitty_32.png',
# ]
# test_alphas = [
#     # 0.2,
#     # 0.2,
#     # 0.2,
#     # 0.2,
#     0.2,
# ]


# k = 4 # number of triggers
# trigger_names = [
#     'firefox_corner_split_1_32.png',
#     'firefox_corner_split_2_32.png',
#     'firefox_corner_split_3_32.png',
#     'firefox_corner_split_4_32.png',
# ]
# alphas = [
#     1,
#     1,
#     1,
#     1,
# ]

# test_k = 4
# test_trigger_names = [
#     'firefox_corner_split_1_32.png',
#     'firefox_corner_split_2_32.png',
#     'firefox_corner_split_3_32.png',
#     'firefox_corner_split_4_32.png',
# ]

# test_alphas = [
#     1,
#     1,
#     1,
#     1,
# ]


# k = 4  # number of triggers
# trigger_names = [
#     # 'hellokitty_32.png',
#     # 'square_center_32.png',
#     # 'square_corner_32.png',
#     'phoenix_corner_32.png',
#     # 'phoenix_corner2_32.png',
#     # 'watermark_white_32.png',
#     'firefox_corner_32.png',
#     'badnet_patch4_32.png',
#     'trojan_square_32.png',
#     # 'trojan_watermark_32.png'
# ]
# alphas = [
#     # 0.2,
#     0.5,
#     # 0.2,
#     0.2,
#     0.5,
#     0.3,
#     # 0.5
# ]

# test_k = 2
# test_trigger_names = [
#     # 'hellokitty_32.png',
#     # 'square_center_32.png',
#     # 'square_corner_32.png',
#     # 'phoenix_corner_32.png',
#     'phoenix_corner2_32.png',
#     # 'watermark_white_32.png',
#     # 'firefox_corner_32.png',
#     'badnet_patch4_32.png',
#     # 'trojan_square_32.png',
#     # 'trojan_watermark_32.png'
# ]

# test_alphas = [
#     # 0.5,
#     # 0.5,
#     1,
#     1,
# ]


class poison_generator():

    def __init__(self, img_size, trainset, testset, dataset_in_class_order, poison_rate, path, trigger_names, alphas, test_trigger_names, test_alphas, target_class=0, cover_rate=0.01):

        self.img_size = img_size
        self.trainset = trainset
        self.testset = testset
        self.dataset_in_class_order = dataset_in_class_order
        self.poison_rate = poison_rate
        self.path = path  # path to save the dataset
        self.target_class = target_class  # by default : target_class = 0
        self.cover_rate = cover_rate

        # number of images
        self.num_train_img = len(trainset)
        self.num_test_img = len(testset)

        # triggers
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        def get_triggers(trigger_names, trigger_marks, trigger_masks):
            for i in range(len(trigger_names)):
                trigger_path = os.path.join(config.triggers_dir, trigger_names[i])
                trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % trigger_names[i])

                trigger = Image.open(trigger_path).convert("RGB")
                trigger = trigger_transform(trigger)

                if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                    trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                    trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
                else:  # by default, all black pixels are masked with 0's
                    trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                    trigger[2] > 0).float()

                trigger_marks.append(trigger)
                trigger_masks.append(trigger_mask)

                print(f"Trigger #{i}: {trigger_names[i]}")

        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = alphas
        get_triggers(trigger_names, self.trigger_marks, self.trigger_masks)

        self.test_trigger_marks = []
        self.test_trigger_masks = []
        self.test_alphas = test_alphas
        get_triggers(test_trigger_names, self.test_trigger_marks, self.test_trigger_masks)

    def generate_poisoned_training_set(self):

        # random sampling
        id_set = list(range(0, self.num_train_img))
        random.shuffle(id_set)
        num_poison = int(self.num_train_img * self.poison_rate)
        poison_indices = id_set[:num_poison]

        num_cover = int(self.num_train_img * self.cover_rate)
        cover_indices = id_set[num_poison:num_poison + num_cover]  # use **non-overlapping** images to cover

        label_set = []
        k = len(self.trigger_marks)

        # In the original code, the choice between the multiple triggers is random. 
        # However, this is only the case under the assumption that the dataset is in random order.
        # For imagenette, the data samples are ordered according to their label.
        # Consequently, each trigger would only be used for a subset of the classes, and the selection would not be truly random.
        # Therefore, we change the trigger selection method if the dataset is in class order.
        if self.dataset_in_class_order:
            poison_indices_per_trigger = np.array_split(poison_indices, k)
            cover_indices_per_trigger = np.array_split(cover_indices, k)

            for i in range(self.num_train_img):
                img, gt = self.trainset[i]

                # cover image
                for j in range(k):
                    if i in cover_indices_per_trigger[j]:
                        img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                        break

                # poisoned image
                for j in range(k):
                    if i in poison_indices_per_trigger[j]:
                        gt = self.target_class  # change the label to the target class
                        img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                        break

                img_file_name = '%d.png' % i
                img_file_path = os.path.join(self.path, "train", img_file_name)
                save_image(img, img_file_path)
                label_set.append(gt)
        else:
            poison_indices.sort()  # increasing order
            cover_indices.sort()

            pt = 0
            ct = 0
            cnt = 0

            poison_id = []
            cover_id = []

            for i in range(self.num_train_img):
                img, gt = self.trainset[i]

                # cover image
                if ct < num_cover and cover_indices[ct] == i:
                    cover_id.append(cnt)
                    for j in range(k):
                        if ct < (j + 1) * (num_cover / k):
                            img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                            # img[j, :, :] = img[j, :, :] + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j][j, :, :] - img[j, :, :])
                            break
                    ct += 1

                # poisoned image
                if pt < num_poison and poison_indices[pt] == i:
                    poison_id.append(cnt)
                    gt = self.target_class  # change the label to the target class
                    for j in range(k):
                        if pt < (j + 1) * (num_poison / k):
                            img = img + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - img)
                            # img[j, :, :] = img[j, :, :] + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j][j, :, :] - img[j, :, :])
                            break
                    pt += 1

                img_file_name = '%d.png' % cnt
                img_file_path = os.path.join(self.path, "train", img_file_name)
                save_image(img, img_file_path)
                # print('[Generate Poisoned Set] Save %s' % img_file_path)
                label_set.append(gt)
                cnt += 1

            poison_indices = poison_id
            cover_indices = cover_id

        label_set = torch.LongTensor(label_set)
        print("Poison indices:", poison_indices)
        print("Cover indices:", cover_indices)

        # demo
        img, gt = self.trainset[0]
        img = img + self.alphas[0] * self.trigger_masks[0] * (self.trigger_marks[0] - img)
        save_image(img, os.path.join(self.path, f'demo_train.png'))

        return poison_indices, cover_indices, label_set

    def generate_poisoned_testing_set(self):

        cnt = 0

        k = len(self.test_trigger_marks)

        for i in range(self.num_test_img):
            img, gt = self.testset[i]

            # Do not save poisoned images originally from the target class
            if gt == self.target_class:
                continue

            for j in range(k):
                img = img + self.test_alphas[j] * self.test_trigger_masks[j] * (self.test_trigger_marks[j] - img)

            img_file_name = '%d.png' % cnt
            img_file_path = os.path.join(self.path, "test", img_file_name)
            save_image(img, img_file_path)
            # print('[Generate Poisoned Set] Save %s' % img_file_path)
            cnt += 1

        # demo
        img, gt = self.testset[0]
        for j in range(k):
            img = img + self.test_alphas[j] * self.test_trigger_masks[j] * (self.test_trigger_marks[j] - img)
        save_image(img, os.path.join(self.path, f'demo_test.png'))

class poison_transform():

    def __init__(self, img_size, test_trigger_names, test_alphas, target_class=0, denormalizer=None, normalizer=None):

        self.img_size = img_size
        self.target_class = target_class
        self.denormalizer = denormalizer
        self.normalizer = normalizer

        # triggers
        trigger_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self.trigger_marks = []
        self.trigger_masks = []
        self.alphas = []
        for i in range(len(test_trigger_names)):
            trigger_path = os.path.join(config.triggers_dir, test_trigger_names[i])
            trigger_mask_path = os.path.join(config.triggers_dir, 'mask_%s' % test_trigger_names[i])
            trigger = Image.open(trigger_path).convert("RGB")
            trigger = trigger_transform(trigger)
            if os.path.exists(trigger_mask_path):  # if there explicitly exists a trigger mask (with the same name)
                trigger_mask = Image.open(trigger_mask_path).convert("RGB")
                trigger_mask = transforms.ToTensor()(trigger_mask)[0]  # only use 1 channel
            else:  # by default, all black pixels are masked with 0's
                trigger_mask = torch.logical_or(torch.logical_or(trigger[0] > 0, trigger[1] > 0),
                                                trigger[2] > 0).float()

            self.trigger_marks.append(trigger.cuda())
            self.trigger_masks.append(trigger_mask.cuda())
            self.alphas.append(test_alphas[i])

    def transform(self, data, labels, denormalizer=None, normalizer=None):
        data, labels = data.clone(), labels.clone()

        data = self.denormalizer(data)
        for j in range(len(self.trigger_marks)):
            data = data + self.alphas[j] * self.trigger_masks[j] * (self.trigger_marks[j] - data)
        data = self.normalizer(data)
        labels[:] = self.target_class

        # debug
        # from torchvision.utils import save_image
        # save_image(self.denormalizer(data)[2], 'a.png')

        return data, labels