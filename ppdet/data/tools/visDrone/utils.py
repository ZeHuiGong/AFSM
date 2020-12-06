import mmcv
import os
import csv
import numpy as np


def write_result_txt(out_root, outputs, imgs_path=None):
    mmcv.mkdir_or_exist(out_root)
    results = outputs['results']
    im_names = outputs['im_names']
    for i, name in enumerate(im_names):  # loop through images
        res = results[i]
        with open(os.path.join(out_root, name[:-3] + 'txt'), 'w') as f:
            for cls, bbox in enumerate(res):  # loop through cls
                if bbox.shape[0] == 0:
                    continue
                bbox = bbox.astype(np.int)
                for obj_i in range(bbox.shape[0]):   # loop through cls-box
                    bbox_left = bbox[obj_i, 0]
                    bbox_top = bbox[obj_i, 1]
                    bbox_width = bbox[obj_i, 2] - bbox[obj_i, 0]
                    bbox_height = bbox[obj_i, 3] - bbox[obj_i, 1]
                    score = bbox[obj_i, 4]
                    line = ','.join(list(map(str, [bbox_left, bbox_top, bbox_width, bbox_height, score, cls+1, -1, -1])))
                    line += '\n'
                    f.write(line)
    if imgs_path is not None:
        all_ims = os.listdir(imgs_path)
        no_obj_ims = list(set(all_ims) - set(im_names))
        for name in no_obj_ims:
            open(os.path.join(out_root, name[:-3] + 'txt'), "w")


def write_f1_tofile(metric, epoch, filename):
    with open(filename, 'a') as f:
        writer = csv.writer(f)
        if isinstance(metric, list):
            writer.writerow([epoch, *metric])
        else:
            writer.writerow([epoch, metric])


def colormap(rgb=False):
    color_list = np.array(
        [
            0.000, 0.447, 0.741,
            0.850, 0.325, 0.098,
            0.929, 0.694, 0.125,
            0.494, 0.184, 0.556,
            0.466, 0.674, 0.188,
            0.301, 0.745, 0.933,
            0.635, 0.078, 0.184,
            0.300, 0.300, 0.300,
            0.600, 0.600, 0.600,
            1.000, 0.000, 0.000,
            1.000, 0.500, 0.000,
            0.749, 0.749, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 1.000,
            0.667, 0.000, 1.000,
            0.333, 0.333, 0.000,
            0.333, 0.667, 0.000,
            0.333, 1.000, 0.000,
            0.667, 0.333, 0.000,
            0.667, 0.667, 0.000,
            0.667, 1.000, 0.000,
            1.000, 0.333, 0.000,
            1.000, 0.667, 0.000,
            1.000, 1.000, 0.000,
            0.000, 0.333, 0.500,
            0.000, 0.667, 0.500,
            0.000, 1.000, 0.500,
            0.333, 0.000, 0.500,
            0.333, 0.333, 0.500,
            0.333, 0.667, 0.500,
            0.333, 1.000, 0.500,
            0.667, 0.000, 0.500,
            0.667, 0.333, 0.500,
            0.667, 0.667, 0.500,
            0.667, 1.000, 0.500,
            1.000, 0.000, 0.500,
            1.000, 0.333, 0.500,
            1.000, 0.667, 0.500,
            1.000, 1.000, 0.500,
            0.000, 0.333, 1.000,
            0.000, 0.667, 1.000,
            0.000, 1.000, 1.000,
            0.333, 0.000, 1.000,
            0.333, 0.333, 1.000,
            0.333, 0.667, 1.000,
            0.333, 1.000, 1.000,
            0.667, 0.000, 1.000,
            0.667, 0.333, 1.000,
            0.667, 0.667, 1.000,
            0.667, 1.000, 1.000,
            1.000, 0.000, 1.000,
            1.000, 0.333, 1.000,
            1.000, 0.667, 1.000,
            0.167, 0.000, 0.000,
            0.333, 0.000, 0.000,
            0.500, 0.000, 0.000,
            0.667, 0.000, 0.000,
            0.833, 0.000, 0.000,
            1.000, 0.000, 0.000,
            0.000, 0.167, 0.000,
            0.000, 0.333, 0.000,
            0.000, 0.500, 0.000,
            0.000, 0.667, 0.000,
            0.000, 0.833, 0.000,
            0.000, 1.000, 0.000,
            0.000, 0.000, 0.167,
            0.000, 0.000, 0.333,
            0.000, 0.000, 0.500,
            0.000, 0.000, 0.667,
            0.000, 0.000, 0.833,
            0.000, 0.000, 1.000,
            0.000, 0.000, 0.000,
            0.143, 0.143, 0.143,
            0.286, 0.286, 0.286,
            0.429, 0.429, 0.429,
            0.571, 0.571, 0.571,
            0.714, 0.714, 0.714,
            0.857, 0.857, 0.857,
            1.000, 1.000, 1.000
        ]
    ).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list
