import mmcv
import numpy as np
import matplotlib.pyplot as plt


def analise_box_cls(classes, num_cls):
    """count the number of objects in each class.
    @:param
    classes: list[nd.narray] or numpy array
    """
    if isinstance(classes, list):
        classes = np.hstack(classes)
    cls_num = []
    print('total number of box:', classes.shape[0])
    for i in range(num_cls):
        cls_num.append(np.where(classes == i + 1)[0].shape[0])
    index = list(range(1, num_cls + 1))
    plt.bar(index, cls_num)
    plt.title('False cls analysis')
    plt.xlabel('cls')
    plt.ylabel('num')
    plt.xticks(index, ('cls{}'.format(i+1) for i in range(num_cls)))
    plt.show()


def analise_box_area(bboxes, bins=20):
    """ analise the distribution of box area(sqrt(wh))
        bboxes: list of numpy array
    """
    if len(bboxes) == 0:
        return
    if isinstance(bboxes, list):
        bboxes = np.vstack(bboxes)

    wh = bboxes[:, 2:4] - bboxes[:, 0:2]
    areas = np.sqrt(wh[:, 0] * wh[:, 1])
    max_area = areas.max()
    edges = [float(x) / bins * max_area for x in range(bins + 1)]
    edges[-1] += 1e-6
    num_dis = [0 for _ in range(bins)]
    for i in range(bins):
        inds = np.where((areas >= edges[i]) &
                        (areas < edges[i+1]))[0]
        num_dis[i] += inds.shape[0]
    max_num = max(num_dis)
    idx = num_dis.index(max_num)
    print('num:', num_dis)
    plt.plot(edges[:-1], num_dis, 'o')
    plt.plot([edges[idx], edges[idx]], [0, max_num], 'r--')
    plt.title('boxes area analysis')
    plt.xlabel('sqrt(wh)')
    plt.ylabel('num')
    plt.show()


def analysize_visdrone(label_file):
    annotations = mmcv.load(label_file)
    bboxes = []
    labels = []
    for anno in annotations:
        bbox = anno['ann']['bboxes']
        label = anno['ann']['labels']
        bboxes.append(bbox)
        labels.append(label)
    analise_box_cls(labels, num_cls=11)
    # analise_box_area(bboxes, bins=20)


def analysize_NumObj_PerImg(label_file):
    annotations = mmcv.load(label_file)
    num_obj_per_img = []
    mt_500_imgs = []
    for anno in annotations:
        label = anno['ann']['labels']
        num_obj_per_img.append(label.shape[0])
        if label.shape[0] > 300:
            mt_500_imgs.append(anno['filename'])

    plt.plot(np.arange(len(num_obj_per_img)), num_obj_per_img, 'ro')
    plt.title('number of object per-image')
    plt.xlabel('img-idx')
    plt.ylabel('num')
    plt.show()
    print('imgs that more than 500 objects:', mt_500_imgs)


if __name__ == '__main__':
    label_file = '/media/jp/新加卷/ZE_HUI/pytorch_code/mmdetection_ctnet' \
                 '/data/visdrone2019/annotations/visdrone2019-train.pkl'
    analysize_visdrone(label_file)
    # analysize_NumObj_PerImg(label_file)
