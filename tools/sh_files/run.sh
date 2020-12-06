export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=0

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

 echo "infer images"
 python -u tools/infer.py -c configs/anchor_free/centernet/Journal_cfgs/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821.yml \
 --infer_dir=demo/ \
 --output_dir=infer_output/ \
 --draw_threshold=0.3 \
 -o weights=output/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821/model_final

# echo "converting(merging) results from coco format to visdrone format..."
# python -u ppdet/data/tools/visDrone/cvt_res_coco2visdrone.py \
# --res_coco_file_or_dir /home/aistudio/work/PaddleDetection/output/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821/results/valset/bbox_scale1.0.json \
# --out_dir /home/aistudio/work/PaddleDetection/output/cbr50_fpn_ChaV1_GcbMixupCasmGIOU_0821/results/val_txts \
# --anno_file /home/aistudio/work/PaddleDetection/dataset/visdrone/annotations/visdrone2019-val.json \
# --top_K 500

