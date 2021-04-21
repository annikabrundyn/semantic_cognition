import os
import numpy as np
import random


def make_data_matrix(root_dir, imgs_per_item):

    with open(os.path.join(root_dir, 'sem_items.txt'), 'r') as fid:
        names_items = np.array([l.strip().lower() for l in fid.readlines()])
    with open(os.path.join(root_dir, 'sem_relations.txt'), 'r') as fid:
        names_relations = np.array([l.strip().lower() for l in fid.readlines()])
    with open(os.path.join(root_dir, 'sem_attributes.txt'), 'r') as fid:
        names_attributes = np.array([l.strip().lower() for l in fid.readlines()])

    nobj = len(names_items)
    nrel = len(names_relations)
    nattr = len(names_attributes)

    data_matrix = np.loadtxt(os.path.join(root_dir, 'sem_data.txt'))
    item_strings = np.apply_along_axis(lambda v: names_items[v.astype('bool')], 1, data_matrix[:, :nobj])
    data2 = np.concatenate((item_strings, data_matrix[:, nobj:]), axis=1, dtype='object')

    img_matrix = []

    for sample in data2:
        item_name = sample[0]
        rel = np.array(sample[1:1 + nrel], dtype='float32')
        attr = np.array(sample[1 + nrel:], dtype='float32')

        for idx in range(0, imgs_per_item):
            item_img = os.path.join(item_name, f"image_{idx}.jpg")
            matrix_item = [item_name, item_img, rel, attr]
            img_matrix.append(matrix_item)

    return img_matrix


### TODO: make this validation

def train_test_split(samples, test_pcnt, seed):
    random.seed(seed)
    random.shuffle(samples)

    n_test_samples = int(test_pcnt * len(samples))
    test = samples[:n_test_samples]
    train = samples[n_test_samples:]

    return train, test


