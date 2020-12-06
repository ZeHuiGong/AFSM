import os
import mmcv
from collections import defaultdict
import numpy as np


def coco2visdrone_res(coco_res_file, out_root, dataset_path):
    """convert coco result format to visdrone result format
    coco_res : list[dict],each element correspond to one box
    {
        image_name:
        bbox: list (x1, y1, w, h)
        score:
        category_id:ori id in visdrone
    }
    """
    im_names = os.listdir(dataset_path)
    coco_res = mmcv.load(coco_res_file)
    mmcv.mkdir_or_exist(out_root)

    for name in im_names:  # create empty file
        open(os.path.join(out_root, name[:-3] + 'txt'), "w")

    results = defaultdict(list)
    for res in coco_res:
        im_name = res['image_name']
        box = res['bbox']
        box.extend([res['score'], res['category_id'], -1, -1])
        results[im_name].append(box)

    for im_name, im_boxes in results.items():  # loop through images
        # img = mmcv.imread(os.path.join(dataset_path, im_name))
        # box = np.array(im_boxes)
        # print('total detected box num:', box.shape[0])
        # box[:, 2] = box[:, 0] + box[:, 2]
        # box[:, 3] = box[:, 1] + box[:, 3]
        # box = box[box[:, 4] > 0.3]
        # mmcv.imshow_det_bboxes(
        #     img,
        #     box[:, 0:4],
        #     labels=box[:, 5],
        #     bbox_color='red',
        #     text_color='green',
        #     thickness=1,
        #     show=True,
        #     win_name='',
        #     wait_time=0)

        with open(os.path.join(out_root, im_name[:-3] + 'txt'), 'a') as f:
            for box in im_boxes:
                if box[4] < 0.1:
                    continue
                line = ','.join(list(map(str, box)))
                line += '\n'
                f.write(line)


if __name__ == '__main__':
    coco_res_file = '../../../results.pkl.json'
    out_root = '../../../work_dirs/visdrone2019/centernet_onepoint_hrnet_dcn_gcb-52_1x-06-25/results/test-640'
    dataset_path = '../../../data/visdrone2019/visdrone2019_val_data'
    coco2visdrone_res(coco_res_file, out_root, dataset_path)



