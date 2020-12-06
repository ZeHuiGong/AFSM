export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

# 2020-6-27 exps: giou loss (instead of predicting wh of the box directly, we regress the distance
# from point to four axis-aligned borders, denoted as [lx, ty, rx, by].)
# configuration: ResNet50 + bifpn + fuse_features + GIOULoss * 1, max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511)

# 2020-6-28 exps: giou loss, only the positive positions, i.e., center points, are considered to compute loss during training.
# configuration: ResNet50 + bifpn + fuse_features + GIOULoss * 5, max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [14:00]

# 2020-6-28 exps: giou loss, only the positive positions, i.e., center points, are considered to compute loss during training.
# configuration: ResNet50 + bifpn + fuse_features + GIOULoss * 5, max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [14:00]

# 2020-6-28 exps: diou loss, only the positive positions, i.e., center points, are considered to compute loss during training.
# configuration: ResNet50 + bifpn + fuse_features + DIOULoss * 0.3, max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [23:20]

# 2020-6-29 exps: 
# ResNet50 + bifpn + fuse_features + L1Loss * 0.1 + class_aware_sampling +mixup(mix_epoch=180), 
# max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [13:40]

# 2020-6-30 exps: 
# ResNet50 + bifpn + fuse_features + (GIOULoss+CIOU_Term) * 1, max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [13:10]

# 2020-7-03 exps: 
# ResNet50 + bifpn + fuse_features + L1loss * 0.1 + !!feature_adaptation!!, 
# max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [18:15]

# 2020-7-04 exps: 
# ResNet50 + bifpn + fuse_features + L1loss * 0.1 + !!mixup(mixup_epoch=180, total_epoch=247, weighted sum loss)!!, 
# max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [14:15] [test_dev: 22.1, val:25.4]

# 2020-7-05 exps:
# 1.ResNet50 + bifpn + fuse_features + L1loss * 0.1 + !!mixup(mixup_epoch=300, mixup_prob=0.6, weighted sum loss)!!, 
# max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [2:50]

# 2.ResNet50 + bifpn + fuse_features + L1loss * 0.1 + !!sac_conv [switch[sigmoid] stages: 4, 5]!!, 
# max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [20:40]

# 2020-08-17
# 1.ResNet50 + fpn + single_scale + ClassAwareSamp + L1loss * 0.1; max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [17:00]

# 2020-08-18
# 1.ResNet50 + fpn + single_scale + GCB(st3->5) + mixup(mixup_epoch=180) + ClassAwareSamp + GIOU * 1.0; 
# max_iters=50000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [15:15]

# 2020-08-21
# [CornerNet + VOC] 1.ResNet101 + fpn + single_scale; max_iters=70000, lr=1.25e-4, bs=16 (w, h) = (511, 511) [0:55]

# 2020-11-18 VocDataset
# 1.R101-vd + fpn + ChaFuseV1 + GCB(st3->5) + mixup(mixup_epoch=85) + ClassAwareSamp + GIOU * 1.0; 
# steplr(22500, 30000); max_iters=35000, lr=1.25e-4, bs=32 (w, h) = (511, 511) [2:15]

#echo "training images"
# python -u tools/train.py -c configs/anchor_free/centernet/Journal_cfgs/voc_exps/r101_vd_fpn_GcbMixupCasmGIOU_1117.yml

# echo "eval images"
for scale in 1.0;do  # 1.25 1.5 1.8
    echo eval with scale ${scale}
    python -u tools/eval.py -c configs/anchor_free/centernet/Journal_cfgs/voc_exps/r101_vd_fpn_GcbMixupCasmGIOU_1117.yml \
    -o weights=output/r101_vd_fpn_GcbMixupCasmGIOU_1117/model_final \
    --output_eval output/r101_vd_fpn_GcbMixupCasmGIOU_1117/results/valset \
    --test_scales ${scale}
done
# --json_eval

# echo "infer images"
# python -u tools/infer.py -c configs/anchor_free/centernet/Journal_cfgs/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821.yml \
# --infer_dir=demo/ \
# --output_dir=infer_output/ \
# --draw_threshold=0.3 \
# -o weights=output/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821/model_final

# echo "converting(merging) results from coco format to visdrone format..."
# python -u ppdet/data/tools/visDrone/cvt_res_coco2visdrone.py \
# --res_coco_file_or_dir /home/aistudio/work/PaddleDetection/output/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821/results/valset/bbox_scale1.0.json \
# --out_dir /home/aistudio/work/PaddleDetection/output/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821/results/val_txts \
# --anno_file /home/aistudio/work/PaddleDetection/dataset/visdrone/annotations/visdrone2019-val.json \
# --top_K 500

