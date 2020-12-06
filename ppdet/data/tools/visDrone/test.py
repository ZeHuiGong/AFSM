import argparse

import torch
import mmcv
from mmcv.runner import load_checkpoint, parallel_test, obj_from_dict
from mmcv.parallel import scatter, collate, MMDataParallel

from mmdet import datasets
from mmdet.datasets import build_dataloader
from mmdet.models import build_detector, detectors
import os
from tools.convert_datasets.visDrone.utils import write_result_txt
from tools.convert_datasets.visDrone.visdrone_eval import visdrone_eval


def single_test(model, data_loader, show=False):
    model.eval()
    results = []
    im_names = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=not show, **data)
        im_names.append(dataset.img_infos[i]['filename'])  # 前提是测试的batch_size 必须为1

        if show:
            model.module.show_result(data, result, dataset.img_norm_cfg,
                                     dataset=dataset.CLASSES)
        results.append(result)
        batch_size = data['img'][0].size(0)
        for _ in range(batch_size):
            prog_bar.update()
    return dict(results=results, im_names=im_names)


def _data_func(data, device_id):
    data = scatter(collate([data], samples_per_gpu=1), [device_id])[0]
    return dict(return_loss=False, rescale=True, **data)


def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test detector')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--gpus', default=1, type=int, help='GPU number used for testing')
    parser.add_argument(
        '--proc_per_gpu',
        default=1,
        type=int,
        help='Number of processes per GPU')
    parser.add_argument('--out', help='output result file')
    parser.add_argument('--val', action='store_true', help='eval mode')
    parser.add_argument('--show', action='store_true', help='show results')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    mmcv.mkdir_or_exist(os.path.join(cfg.work_dir, 'results'))

    print(args)
    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.model.pretrained = None
    cfg.data.test.test_mode = True

    if args.val:
        print('now is val mode, with label')
        imgs_path = 'data/visdrone2019/visdrone2019_val_data'
        dataset = obj_from_dict(cfg.data.val, datasets, dict(test_mode=True))
    else:
        print('now is test mode, no label')
        imgs_path = 'data/visdrone2019/visdrone2019-test_data'
        dataset = obj_from_dict(cfg.data.test, datasets, dict(test_mode=True))
    if args.gpus == 1:
        model = build_detector(
            cfg.model, train_cfg=None, test_cfg=cfg.test_cfg)
        load_checkpoint(model, args.checkpoint)
        model = MMDataParallel(model, device_ids=[0])

        data_loader = build_dataloader(
            dataset,
            imgs_per_gpu=1,
            workers_per_gpu=cfg.data.workers_per_gpu,
            num_gpus=1,
            dist=False,
            shuffle=False)
        outputs = single_test(model, data_loader, args.show)
    else:
        model_args = cfg.model.copy()
        model_args.update(train_cfg=None, test_cfg=cfg.test_cfg)
        model_type = getattr(detectors, model_args.pop('type'))
        outputs = parallel_test(
            model_type,
            model_args,
            args.checkpoint,
            dataset,
            _data_func,
            range(args.gpus),
            workers_per_gpu=args.proc_per_gpu)

    if args.out:
        print('writing results to {}'.format(args.out))
        write_result_txt(args.out, outputs, imgs_path)
        if args.val:  # eval
            visdrone_eval(args.out, dataset.gtPath, dataset.ann_file)


if __name__ == '__main__':
    main()
