import os
import numpy as np


def make_data_matrix(root_dir, imgs_per_item):

    with open(os.path.join(root_dir, 'sem_items.txt'), 'r') as fid:
        names_items = np.array([l.strip().lower() for l in fid.readlines()])
    with open(os.path.join(root_dir, 'sem_relations.txt'), 'r') as fid:
        names_relations = np.array([l.strip().lower() for l in fid.readlines()])
    with open(os.path.join(root_dir, 'sem_attributes.txt'), 'r') as fid:
        names_attributes = np.array([l.strip().lower() for l in fid.readlines()])

    nobj = len(names_items)
    nrel = len(names_relations)

    data_matrix = np.loadtxt(os.path.join(root_dir, 'sem_data.txt'))

    item_strings = np.apply_along_axis(lambda v: names_items[v.astype('bool')], 1, data_matrix[:, :nobj])

    data2 = np.concatenate((item_strings, data_matrix[:, nobj:]), axis=1, dtype='object')

    img_matrix = []

    for sample in data2:
        item_name = sample[0]
        rel = list(sample[1:1 + nrel])
        attr = list(sample[1 + nrel:])

        for idx in range(1, imgs_per_item + 1):
            item_img = os.path.join(item_name, f"Image_{idx}")
            matrix_item = [item_name, item_img, rel, attr]
            img_matrix.append(matrix_item)

    return img_matrix