import random
import mmcv
from PIL import Image
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import os
import copy
import numpy as np
from mmdet.datasets.my_dataset import vis_bbox


def data_patch_aug(dataset_path, annotation_path, out_data_path, out_anno_path):
    mmcv.mkdir_or_exist(out_data_path)
    annotations = mmcv.load(annotation_path)
    new_annotations = []
    total_ImNum = len(annotations)
    thereshold = 0.15
    specific_aug_cls = {2: 1, 5: 1, 7: 2, 8: 2, 9: 1}
    for i, annos in enumerate(annotations):  # loop throug images
        print('augmenting {}/{}'.format(i+1, total_ImNum))
        im_path = os.path.join(dataset_path, annotations[i]['filename'])
        img = Image.open(im_path)
        new_instance = copy.copy(annos)

        num_obj_this_img = annos['ann']['bboxes'].shape[0]
        for obj_i in range(num_obj_this_img):  # loop through objs of the image
            cls_obj = annos['ann']['labels'][obj_i]
            if cls_obj not in specific_aug_cls.keys():
                continue
            sample_num_this_obj = specific_aug_cls[cls_obj]
            new_boxes = generate_new_box(
                annos["width"], annos['height'],
                annos['ann']['bboxes'][obj_i],
                sample_num_this_obj,
                new_instance['ann']['bboxes'],
                iou_thr=thereshold)
            new_instance['ann']['bboxes'] = np.append(new_instance['ann']['bboxes'], new_boxes, axis=0)
            labels = np.empty(sample_num_this_obj, dtype=np.int64)
            labels.fill(annos['ann']['labels'][obj_i])
            new_instance['ann']['labels'] = np.append(new_instance['ann']['labels'], labels, axis=0)
            crop_and_paste(annos['ann']['bboxes'][obj_i], new_boxes, img)
        new_annotations.append(new_instance)
        img.save(os.path.join(out_data_path, annotations[i]['filename']))

        # vis_bbox(np.array(img), new_instance['ann']['bboxes'], new_instance['ann']['labels'])
        # if i > 10:
        #     break
    print('aug done!\nwriting annotations to {}'.format(out_anno_path))
    mmcv.dump(new_annotations, out_anno_path)


def generate_new_box(im_width, im_height, copy_box, num_boxes, boxes_set, iou_thr):
    """
    :param copy_box:the box that need to be copy and patch ndarray [5]
                    (x1, y1, x2, y2)
    :param num_boxes: number of copy_box to be copied
    :param boxes_set: all boxes in the specific img
    :param iou_thr: a threshold of controlling the occlusion between
                    new_box and existing boxes_set
    """
    w, h = copy_box[2:4] - copy_box[0:2]
    idx = 0
    new_boxes = np.empty([0, 4], dtype=np.float32)
    while idx < num_boxes:
        new_x1 = random.randint(0, im_width - w - 1)
        new_y1 = random.randint(0, im_height - h - 1)
        new_box = np.array([[new_x1, new_y1, new_x1 + w, new_y1 + h]], dtype=np.int)
        ious = bbox_overlaps(new_box, boxes_set)
        max_ious = ious.max(axis=1)
        if max_ious <= iou_thr:
            # this is valid new_box
            new_boxes = np.append(new_boxes, new_box, axis=0)
            idx += 1
    assert new_boxes.shape[0] == num_boxes, 'expected {}, but got {}'.format(
        num_boxes, new_boxes.shape[0])
    return new_boxes


def crop_and_paste(ori_box, boxes, img):
    """
        :param ori_boxes: ndarray(4) original box of a image
        :param boxes: new boxes that generate from ori_box, ndarray(n, 4)
        :param tar_img: the image that img_patch paste on
        :param source_img: the image that img_patch crop from
    """
    region = img.crop(tuple(ori_box))
    boxes = boxes.astype(np.int64)
    for i in range(boxes.shape[0]):
        img.paste(region, tuple(boxes[i, 0:2]))

    return img


if __name__ == '__main__':
    data_path = '/media/jp/新加卷/ZE_HUI/datasets/visdrone/VisDrone2019-DET-train/images'
    annotation_path = '/media/jp/新加卷/ZE_HUI/datasets/visdrone/cvted_annotations/visdrone2019-train.pkl'
    out_data_path = '/media/jp/新加卷/ZE_HUI/datasets/visdrone/VisDrone2019-DET-train_PatchAug'
    out_anno_path = '/media/jp/新加卷/ZE_HUI/datasets/visdrone/cvted_annotations/visdrone2019-train_PatchAug.pkl'
    data_patch_aug(data_path, annotation_path, out_data_path, out_anno_path)
