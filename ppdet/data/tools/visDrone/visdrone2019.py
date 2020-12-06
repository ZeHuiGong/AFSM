import os.path as osp
import os
import mmcv
import numpy as np
import glob
import time
import copy


info = {"description": "COCO 2017 Dataset",
            "url": "http://cocodataset.org",
            "version": "1.0",
            "year": 2017,
            "contributor": "COCO Consortium",
            "date_created": "2017/09/01"}
licenses = [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1,
             "name": "Attribution-NonCommercial-ShareAlike License"},
            {"url": "http://creativecommons.org/licenses/by-nc/2.0/", "id": 2,
             "name": "Attribution-NonCommercial License"},
            {"url": "http://creativecommons.org/licenses/by-nc-nd/2.0/", "id": 3,
             "name": "Attribution-NonCommercial-NoDerivs License"},
            {"url": "http://creativecommons.org/licenses/by/2.0/", "id": 4, "name": "Attribution License"},
            {"url": "http://creativecommons.org/licenses/by-sa/2.0/", "id": 5,
             "name": "Attribution-ShareAlike License"},
            {"url": "http://creativecommons.org/licenses/by-nd/2.0/", "id": 6,
             "name": "Attribution-NoDerivs License"},
            {"url": "http://flickr.com/commons/usage/", "id": 7, "name": "No known copyright restrictions"},
            {"url": "http://www.usa.gov/copyright.shtml", "id": 8, "name": "United States Government Work"}]
categories = [{"supercategory": "people", "id": 1, "name": "pedestrian"},
              {"supercategory": "people", "id": 2, "name": "person"},
              {"supercategory": "vehicle", "id": 3, "name": "bicycle"},
              {"supercategory": "vehicle", "id": 4, "name": "car"},
              {"supercategory": "vehicle", "id": 5, "name": "van"},
              {"supercategory": "vehicle", "id": 6, "name": "truck"},
              {"supercategory": "vehicle", "id": 7, "name": "tricycle"},
              {"supercategory": "vehicle", "id": 8, "name": "awning-tricycle"},
              {"supercategory": "vehicle", "id": 9, "name": "bus"},
              {"supercategory": "vehicle", "id": 10, "name": "motor"}]


def loadtxt(txt_file):
    lines = mmcv.list_from_file(txt_file)
    if len(lines) == 0:
        return np.zeros([0, 8], dtype=np.int)
    lines = [list(map(int, line.split(',')[:-1])) for line in lines]
    lines = np.array(lines, dtype=np.int)
    return lines


def parse_txt(args):
    txt_path, img_path = args
    img = mmcv.imread(img_path)
    h, w = img.shape[:2]
    annotation = mmcv.list_from_file(txt_path)
    bboxes = []
    labels = []
    for obj in annotation:
        obj = obj.split(',')
        label = int(obj[5])
        if 0 < label < 11:  # if label is 0,it will be ignore
            bbox = [
                int(obj[0]),
                int(obj[1]),
                int(obj[0]) + int(obj[2]),
                int(obj[1]) + int(obj[3]),
            ]
            bboxes.append(bbox)
            labels.append(label)
    if not bboxes:
        bboxes = np.zeros((0, 4))
        labels = np.zeros((0, ))
    else:
        bboxes = np.array(bboxes, ndmin=2)
        labels = np.array(labels)

    bboxes_ignore = np.zeros((0, 4))
    labels_ignore = np.zeros((0,))

    annotation = {
        'filename': img_path.split('/')[-1],
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations_custom(data_path, out_file):
    img_root_path = osp.join(data_path, 'images')
    label_root_path = osp.join(data_path, 'annotations')

    img_names = os.listdir(img_root_path)
    txt_paths = [
        osp.join(label_root_path, img_name[:-3] + 'txt')
        for img_name in img_names
    ]
    img_paths = [
        osp.join(img_root_path, img_name)
        for img_name in img_names
    ]
    annotations = mmcv.track_progress(parse_txt,
                                      list(zip(txt_paths, img_paths)))
    mmcv.dump(annotations, out_file)


def bbox2mask(bbox):
    area = int(bbox[2] * bbox[3])

    polys = [bbox[0], bbox[1],  # tl
             bbox[0]+bbox[2], bbox[1],
             bbox[0]+bbox[2], bbox[1]+bbox[3],
             bbox[0], bbox[1]+bbox[3]
            ]
    polys = list(map(int, polys))
    return [polys], area


def cvt_annotations_coco(data_path, out_file):
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

    img_root_path = osp.join(data_path, 'images')
    label_root_path = osp.join(data_path, 'annotations')
    img_names = os.listdir(img_root_path)
    cats = list(range(1, 11))
    num_anno = 0
    # box_min_size = 100
    for img_i, im_name in enumerate(img_names):  # loop through images
        # print('{}/{}'.format(img_i + 1, 6470))
        img = mmcv.imread(os.path.join(img_root_path, im_name))
        h, w = img.shape[:2]
        image['id'] = img_i + 1
        image['width'] = w
        image['height'] = h
        image['file_name'] = im_name
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        image['date_captured'] = time_now
        images.append(copy.deepcopy(image))

        txt_anno = loadtxt(os.path.join(label_root_path, im_name[:-3] + 'txt'))
        for ann_i in range(txt_anno.shape[0]):
            # area = txt_anno[ann_i, 2] * txt_anno[ann_i, 3]
            if txt_anno[ann_i, 5] not in cats:
                continue
            num_anno += 1
            annotation['id'] = num_anno
            annotation['image_id'] = img_i + 1
            annotation['category_id'] = int(txt_anno[ann_i, 5])
            annotation['iscrowd'] = int(txt_anno[ann_i, 4] == 0)
            annotation['segmentation'], annotation['area'] = bbox2mask(txt_anno[ann_i, :4])
            annotation['bbox'] = list(map(int, txt_anno[ann_i, :4].tolist()))
            annotations.append(copy.deepcopy(annotation))
            annotation['segmentation'].clear()

    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['categories'] = categories
    coco_annotations['annotations'] = annotations
    coco_annotations['images'] = images

    print('saving coco annotations')
    mmcv.dump(coco_annotations, out_file)
    print('done!')


def get_test_annoCoco(test_data_path, out_file):
    img_names = glob.iglob(os.path.join(test_data_path, 'image/*'))
    coco_annotations = {}
    image = {"id": 0,
             'width': 0,
             'height': 0,
             'file_name': '',
             'license': 1,
             'flickr_url': '',
             'coco_url': '',
             'date_captured': ''}
    images = []
    for img_i, im_name in enumerate(img_names):
        im = mmcv.imread(im_name)
        h, w = im.shape[:2]
        image['id'] = img_i + 20000
        image['width'] = w
        image['height'] = h
        image['file_name'] = im_name.split('/')[-1]
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        image['date_captured'] = time_now
        images.append(copy.deepcopy(image))
    coco_annotations['info'] = info
    coco_annotations['licenses'] = licenses
    coco_annotations['categories'] = categories
    coco_annotations['annotations'] = []
    coco_annotations['images'] = images
    
    print('saving coco annotations')
    mmcv.dump(coco_annotations, out_file)
    
    
cvt_func = dict(
    custom_train=cvt_annotations_custom,
    custom_val=cvt_annotations_custom,
    # custom_test=get_test_anno,
    coco_train=cvt_annotations_coco,
    coco_val=cvt_annotations_coco,
    coco_test_dev=cvt_annotations_coco,
    coco_test_challenge=get_test_annoCoco
)


def main(ann_type='custom', split='train', data_path='', out_dir=''):
    dataset_path = data_path.format(split)
    out_file = osp.join(out_dir, 'visdrone2019-{}.{}'.format(split, 'pkl' if ann_type in 'custom' else 'json'))
    
    mmcv.mkdir_or_exist(out_dir)
    
    print('start converting...')
    cvt_func_type = '{}_{}'.format(ann_type, split)
    cvt_func[cvt_func_type](dataset_path, out_file)
    print('done!')


def filter_label(label_file, ignore_ims):
    ori_annotations = mmcv.load(label_file)
    new_annotations = []
    for anno in ori_annotations:
        if anno['filename'] not in ignore_ims:
            new_annotations.append(anno)
    mmcv.dump(new_annotations, label_file)


if __name__ == '__main__':
    data_path = '/media/jp/187E92D97E92AF4E/ZE_HUI/datasets/Visdrone2020/VisDrone2019-DET-{}'
    out_dir = '/media/jp/187E92D97E92AF4E/ZE_HUI/datasets/Visdrone2020/annotations'
    main(ann_type='coco', split='test_challenge',
         data_path=data_path,
         out_dir=out_dir)

    # label_file = '../../../data/visdrone2019/annotations/visdrone2019-train.pkl'
    # filter_label(label_file, ['0000059_01886_d_0000114.jpg'])
