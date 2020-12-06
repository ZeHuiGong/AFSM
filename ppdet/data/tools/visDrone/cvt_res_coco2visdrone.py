import mmcv
import os
import argparse
from pycocotools.coco import COCO
import glob
import numpy as np

import sys
parent_path = os.path.abspath(os.path.join(__file__, *(['..'] * 5)))
if parent_path not in sys.path:
    sys.path.append(parent_path)
from ppdet.utils.post_process import get_nms_result


def main():
    """
    convert coco format json result file to visdrone txt result files
    """
    nms_cfg = dict(
        use_soft_nms=True,
        detections_per_im=FLAGS.top_K,
        nms_thresh=0.001,
        sigma=0.5,)
        # enable_voting=True)
    coco_gt = COCO(FLAGS.anno_file)
    image_ids = coco_gt.getImgIds()
    image_id2name = {img_id: coco_gt.loadImgs(ids=[img_id])[0]['file_name']
                     for img_id in image_ids}
    print('loading results...')
    image_results, need_nms = load_coco_results(FLAGS.res_coco_file_or_dir, image_ids)
    print('executing nms...')
    image_results = nms_results(image_results, nms_cfg, FLAGS.num_classes, need_nms)
    # image_esults: dict[np.ndarray]
    print('saving results to visdrone format')
    mmcv.mkdir_or_exist(FLAGS.out_dir)
    for img_id, res in image_results.items():  # loop through images
        im_name = image_id2name[img_id]
        with open(os.path.join(FLAGS.out_dir, im_name[:-4] + '.txt'), 'w') as f:
            scores = res[:, 1]
            idxes = np.argsort(-scores)
            res = res[idxes]
            total_boxes = res.shape[0]
            for idx in range(total_boxes):
                box = res[idx]
                cat, score = box[:2]
                x1, y1, x2, y2 = np.round(box[2:]).astype(np.int).tolist()
                line_content = '{},{},{},{},{},{},{},{}{}'.format(
                    x1, y1, x2-x1, y2-y1, score, int(cat) + 1, -1, -1,
                    '\n' if idx < total_boxes-1 else '')
                f.write(line_content)

    # optionally, saving the results in coco format,
    # for further evaluation (aug testing case )
    if FLAGS.json_out_file:
        print('saving results to coco evaluation format...')
        save_json_results(image_results, FLAGS.num_classes, FLAGS.json_out_file)


def load_coco_results(file_or_dir, image_ids):
    """load coco json results from json file or the directory
       that contain many json files(augmentation testing)"""
    image_results = {img_id: [] for img_id in image_ids}
    need_nms = False
    if os.path.isdir(file_or_dir):
        need_nms = True
        for file in glob.iglob(file_or_dir + '/*.json'):
            results = mmcv.load(file)
            [image_results[box_info['image_id']].append(box_info)
             for box_info in results]
    else:
        results = mmcv.load(file_or_dir)
        [image_results[box_info['image_id']].append(box_info)
         for box_info in results]
    return image_results, need_nms


def nms_results(results, nms_cfg, num_classes, need_nms=False):
    """results (dict[list[dict]]):
    need_nms (bool): to get a unified format to simplify post process, for need_nms=False,
    we change the format as after nms
    cls_boxes (np.ndarray): [n, 6] (cls, score, x1, y1, x2, y2)
    """
    nms_results = {}
    for img_id, img_res in results.items():
        # (x1, y1, w, h)
        boxes = np.array([box_info['bbox'] for box_info in img_res], dtype=np.float32)
        boxes[:, 2:4] = boxes[:, 0:2] + boxes[:, 2:4]
        scores = np.array([box_info['score'] for box_info in img_res], dtype=np.float32)
        labels = np.array([box_info['category_id'] for box_info in img_res], dtype=np.int)
        labels = labels - 1
        if need_nms:
            cls_boxes = get_nms_result(boxes, scores, nms_cfg, num_classes, background_label=-1, labels=labels)
        else:
            cls_boxes = np.hstack(
                (labels[:, np.newaxis], scores[:, np.newaxis], boxes)).astype(np.float32, copy=False)
        nms_results[img_id] = cls_boxes

    return nms_results


def save_json_results(results, num_classes, out_file):
    xywh_res = []
    clsid2catid = {cls: cls + 1 for cls in range(num_classes)}
    for img_id, res in results.items():  # loop through images
        for k in range(res.shape[0]):    # loop through bounding boxes
            dt = res[k]
            clsid, score, xmin, ymin, xmax, ymax = dt.tolist()
            catid = (clsid2catid[int(clsid)])
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            coco_res = {
                'image_id': img_id,
                'category_id': catid,
                'bbox': bbox,
                'score': score
            }
            xywh_res.append(coco_res)
    assert out_file.endswith('.json')
    mmcv.dump(xywh_res, out_file)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--res_coco_file_or_dir',
        type=str,
        default='',
        help='path to load coco format result file(s).')
    parser.add_argument(
        '--out_dir',
        type=str,
        default='visdrone_out',
        help='directory to save converted visdrone format out txt files.'
    )
    parser.add_argument(
        '--anno_file',
        type=str,
        help='path to load coco format annotations.'
    )
    parser.add_argument(
        '--json_out_file',
        type=str,
        default=None,
        help='if it is not None, then, save the nms results back '
             'to coco json format for further evaluation.'
    )
    parser.add_argument(
        '--num_classes',
        type=int,
        default=10,
        help='number of classes.'
    )
    parser.add_argument(
        '--top_K',
        type=int,
        default=500,
        help='max number of objects in one images'
    )
    FLAGS = parser.parse_args()
    main()
