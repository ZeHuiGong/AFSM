import torch
from mmdet.models import CenterNetHead
from mmdet.datasets.my_dataset import vis_bbox
from mmdet.datasets.loader.build_loader import build_dataloader
import mmcv
from mmdet.core import tensor2imgs
from mmdet.datasets import get_dataset


def gt2out(gt_bboxes_list, gt_labels_list, inp_shapes_list, stride, categories):
    """transform ground truth into output format"""
    batch_size = len(gt_bboxes_list)
    inp_shapes = gt_bboxes_list[0].new_tensor(inp_shapes_list, dtype=torch.int)
    output_size = inp_shapes[0] / stride
    height_ratio, width_ratio = output_size.float() / inp_shapes[0].float()

    # allocating memory
    tl_heatmaps = -2 * gt_bboxes_list[0].new_ones(batch_size, categories, output_size[0], output_size[1])
    br_heatmaps = -2 * gt_bboxes_list[0].new_ones(batch_size, categories, output_size[0], output_size[1])
    ct_heatmaps = -2 * gt_bboxes_list[0].new_ones(batch_size, categories, output_size[0], output_size[1])
    tl_regrs = gt_bboxes_list[0].new_zeros(batch_size, 2, output_size[0], output_size[1])
    br_regrs = gt_bboxes_list[0].new_zeros(batch_size, 2, output_size[0], output_size[1])
    ct_regrs = gt_bboxes_list[0].new_zeros(batch_size, 2, output_size[0], output_size[1])
    tl_emds = gt_labels_list[0].new_zeros(batch_size, 1, output_size[0], output_size[1])
    br_emds = gt_labels_list[0].new_zeros(batch_size, 1, output_size[0], output_size[1])

    for b_ind in range(batch_size):  # loop through batch-images
        for obj_ind, detection in enumerate(gt_bboxes_list[b_ind]):  # loop through objects in one image
            category = gt_labels_list[b_ind][obj_ind] - 1
            xtl, ytl = detection[0], detection[1]
            xbr, ybr = detection[2], detection[3]
            xct, yct = (detection[2] + detection[0]) / 2., (detection[3] + detection[1]) / 2.

            fxtl = (xtl * width_ratio)
            fytl = (ytl * height_ratio)
            fxbr = (xbr * width_ratio)
            fybr = (ybr * height_ratio)
            fxct = (xct * width_ratio)
            fyct = (yct * height_ratio)

            xtl = int(fxtl)
            ytl = int(fytl)
            xbr = int(fxbr)
            ybr = int(fybr)
            xct = int(fxct)
            yct = int(fyct)

            # heatmaps
            tl_heatmaps[b_ind, category, ytl, xtl] = 1
            br_heatmaps[b_ind, category, ybr, xbr] = 1
            ct_heatmaps[b_ind, category, yct, xct] = 1

            # offsets
            tl_regrs[b_ind, 0, ytl, xtl] = fxtl - xtl  # tl_tx
            tl_regrs[b_ind, 1, ytl, xtl] = fytl - ytl  # tl_ty
            br_regrs[b_ind, 0, ybr, xbr] = fxbr - xbr  # br_tx
            br_regrs[b_ind, 1, ybr, xbr] = fybr - ybr  # br_ty
            ct_regrs[b_ind, 0, yct, xct] = fxct - xct  # ct_tx
            ct_regrs[b_ind, 1, yct, xct] = fyct - yct  # ct_ty

            # embeddings
            tl_emds[b_ind, 0, ytl, xtl] = 2
            br_emds[b_ind, 0, ybr, xbr] = 2

    tl_out=(tl_heatmaps, tl_emds, tl_regrs)
    br_out=(br_heatmaps, br_emds, br_regrs)
    ct_out=(ct_heatmaps, None, ct_regrs)

    return tl_out, br_out, ct_out


def out2box(outs, img_meta, num_clses):
    """transform output format into final detection results"""
    decode_cfg = dict(
        K=100,
        kernel=3,
        ae_threshold=0.5,
        num_dets=1000)
    ct_cfg = dict(
        score_thr=0.05,
        nms=dict(type='soft_nms', iou_thr=0.5, min_score=0.05),
        max_per_img=100)
    head = CenterNetHead(in_channels=1, inner_channels=1, num_classes=num_clses)
    det_bboxes, det_labels = head.get_det_bboxes(
        *outs, img_meta, decode_cfg, rescale=False, cfg=ct_cfg)
    bboxes = det_bboxes.numpy()
    labels = det_labels.numpy()

    return bboxes, labels


def main(cfg_file, test_num=1):
    """ data_path: path to images
        label_path: path to annotations
        idxes: index of image is going to be tested with output process
    """
    cfg = mmcv.Config.fromfile(cfg_file)
    dataset = get_dataset(cfg.data.val)

    data_loader = build_dataloader(
        dataset,
        imgs_per_gpu=1,
        workers_per_gpu=1,
        num_gpus=1,
        dist=True,
        shuffle=False)
    for i, data in enumerate(data_loader):
        imgs = tensor2imgs(data['img'].data[0], **cfg.img_norm_cfg)
        gt_boxes = data['gt_bboxes'].data[0]
        gt_labels = data['gt_labels'].data[0]

        inp_shapes = [meta['pad_shape'][:2] for meta in data['img_meta'].data[0]]
        outs = gt2out(gt_boxes, gt_labels, inp_shapes, stride=4, categories=len(dataset.CLASSES))
        bboxes, labels = out2box(outs, data['img_meta'].data[0], len(dataset.CLASSES))
        vis_bbox(imgs[0], gt_boxes[0].cpu().numpy(),
                 gt_labels[0].cpu().numpy(),
                 show=True, show_str='ground truth')
        print('num detected box:', bboxes.shape[0])
        vis_bbox(imgs[0], bboxes, labels, show=True, show_str='transformed boxes', color='green')

        if i >= test_num:
            break


if __name__ == '__main__':
    """ test whether the output process is right.trun ground truth into output format
        then use the output process to get final detected boxes
    """
    cfg_file = '/media/jp/新加卷/ZEHUI_DATA/pytorch_code/mmdetection/configs/centernet/centernet_hourglass-52_1x.py'
    main(cfg_file, test_num=1)
