import time
import os
import copy
import mmcv
from tools.convert_datasets.visDrone.visdrone2019 import info, licenses, categories, bbox2mask


def custom2coco(cusAnnFile, coco_file):
    coco_annotations = {}
    image = {"id": 0,
             'width': 0,
             'height': 0,
             'file_name': '',
             'license': 1,
             'flickr_url': '',
             'coco_url': '',
             'date_captured': ''}
    annotation = {'id': 0,
                  'image_id': 0,
                  'category_id': 0,
                  'segmentation': [],
                  'area': 0,
                  'bbox': [],
                  'iscrowd': 0}

    images = []
    annotations = []
    num_anno = 0
    box_min_size = 60
    cusAnn = mmcv.load(cusAnnFile)
    for img_i, custom_data in enumerate(cusAnn):  # loop through images
        # print('{}/{}'.format(img_i + 1, 6470))
        image['id'] = img_i + 1
        image['width'] = custom_data['width']
        image['height'] = custom_data['height']
        image['file_name'] = custom_data['filename']
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        image['date_captured'] = time_now
        images.append(copy.deepcopy(image))

        for ann_i in range(custom_data['ann']["bboxes"].shape[0]):  # loop through objs
            box = custom_data['ann']["bboxes"][ann_i]
            cls = custom_data['ann']["labels"][ann_i]
            w = box[2] - box[0]
            h = box[3] - box[1]
            box = [box[0], box[1], w, h]
            if (w * h < box_min_size) or (cls == 11):
                continue
            num_anno += 1
            annotation['id'] = num_anno
            annotation['image_id'] = img_i + 1
            annotation['category_id'] = int(cls)
            annotation['iscrowd'] = 0
            annotation['segmentation'], annotation['area'] = bbox2mask(box)
            annotation['bbox'] = list(map(int, box))
            annotations.append(copy.deepcopy(annotation))
            annotation['segmentation'].clear()

    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['categories'] = categories
    coco_annotations['annotations'] = annotations
    coco_annotations['images'] = images

    print('saving coco annotations')
    mmcv.dump(coco_annotations, coco_file)
    print('done!')


if __name__ == '__main__':
    custom_file = '/media/jp/新加卷/ZE_HUI/datasets/visdrone/cvted_annotations/visdrone2019-train_PatchAug.pkl'
    coco_file = '/media/jp/新加卷/ZE_HUI/datasets/visdrone/cvted_annotations/visdrone2019-train_PatchAug.json'
    custom2coco(custom_file, coco_file)
