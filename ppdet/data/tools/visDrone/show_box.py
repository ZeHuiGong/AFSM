import mmcv
import os.path as osp
from pycocotools.coco import COCO
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt


def show_gt_boxes(data_path, label_path, show_out_path, idxes):
    annotations = mmcv.load(label_path)

    def get_labels_single(idx, annotations):
        if isinstance(idx, str):  # show with im_name
            for ann in annotations:
                if ann['filename'] == idx:
                    boxes = ann['ann']['bboxes']
                    labels = ann['ann']['labels']
                    im_name = ann['filename']
                    return boxes, labels, im_name
        else:
            boxes = annotations[idx]['ann']['bboxes']
            labels = annotations[idx]['ann']['labels']
            im_name = annotations[idx]['filename']
            return boxes, labels, im_name

    for idx in idxes:
        boxes, labels, im_name = get_labels_single(idx, annotations)
        img = mmcv.imread(osp.join(data_path, im_name))
        filename = osp.join(show_out_path, im_name)
        mmcv.imshow_det_bboxes(
            img,
            boxes,
            labels,
            bbox_color='red',
            text_color='red',
            thickness=1,
            show=False,
            win_name='',
            wait_time=0,
            out_file=filename)


def show_coco_annotations(data_path, label_path, idx=None, classi=''):
    coco = COCO(label_path)
    cats = coco.loadCats(coco.getCatIds())
    nms = [cat['name'] for cat in cats]
    if idx is None:
        catIds = coco.getCatIds(catNms=[classi])
        imgIds = coco.getImgIds(catIds=catIds)
    else:
        imgIds = coco.getImgIds(imgIds=[*idx])
    for imgId in imgIds:
        img_info = coco.loadImgs(imgId)[0]
        img = mmcv.imread(osp.join(data_path, img_info['file_name']))
        plt.axis('off')
        plt.imshow(img)
        plt.title(img_info['file_name'])
        annIds = coco.getAnnIds(imgIds=img_info['id'], iscrowd=None)
        anns = coco.loadAnns(annIds)
        coco.showAnns(anns)
        ax = plt.gca()
        for object in anns:
            ax.add_patch(Rectangle((object['bbox'][0], object['bbox'][1]),
                                   object['bbox'][2],
                                   object['bbox'][3],
                                   fill=False,
                                   edgecolor='red' if object["iscrowd"] else 'green',
                                   linewidth=1))
            ax.text(object['bbox'][0], object['bbox'][1], nms[object['category_id'] - 1],
                    style='italic',
                    bbox={'facecolor': 'white', 'alpha': 0.7, 'pad': 0.5})
        plt.show()


if __name__ == '__main__':
    split = 'test_dev'
    label_path = '/media/jp/187E92D97E92AF4E/ZE_HUI/datasets/Visdrone2020/annotations/visdrone2019-{}.json'.format(split)
    data_path = '/media/jp/187E92D97E92AF4E/ZE_HUI/datasets/Visdrone2020/VisDrone2019-DET-{}/images'.format(split)
    show_out_path = '/media/jp/新加卷/ZE_HUI/datasets/ChestCT/demo'

    idxes = list(range(8000, 8030))
    # show_gt_boxes(data_path, label_path, show_out_path, idxes)
    show_coco_annotations(data_path, label_path, idx=None, classi='truck')
