import torch
from mmdet.models import build_detector
import mmcv

cfg_file = '/media/jp/新加卷/ZEHUI_DATA/pytorch_code/mmdetection/configs/centernet/centernet_hourglass-52_1x.py'
pretrained_weight_file = '/media/jp/新加卷/ZEHUI_DATA/权重文件/pre-trained-weights/CenterNet/CenterNet-52_18000.pkl'
out_file = '/media/jp/新加卷/ZEHUI_DATA/权重文件/pre-trained-weights/CenterNet/visdrone-CenterNet-52_18000.pth'

cfg = mmcv.Config.fromfile(cfg_file)
cfg.model.pretrained = None

# construct the model
model = build_detector(cfg.model, test_cfg=cfg.test_cfg)
pretrained_state_dict = torch.load(pretrained_weight_file)
state_dict = model.state_dict()
state_dict_copy = state_dict.copy()

for i, key in enumerate(pretrained_state_dict.keys()):
    print(i+1, key)
count = 0
for (model_key, model_parm), (pre_key, pre_parm) in zip(state_dict.items(), pretrained_state_dict.items()):
    if 'backbone' in model_key and model_parm.shape == pre_parm.shape:
        state_dict_copy[model_key].copy_(pre_parm)
        count += 1

for i, key in enumerate(list(pretrained_state_dict.keys())):
    if i < count:
        pretrained_state_dict.pop(key)
print('left num parm :', len(pretrained_state_dict))

tl_corners = []
br_corners = []
center = []
for key, parm in pretrained_state_dict.items():
    if ('tl_cnvs' in key) or ('tl_heats' in key) or ('tl_tags' in key) or ('tl_regrs' in key):
        tl_corners.append(parm)
    elif ('br_cnvs' in key) or ('br_heats' in key) or ('br_tags' in key) or ('br_regrs' in key):
        br_corners.append(parm)
    elif ('ct_cnvs' in key) or ('ct_heats' in key) or ('ct_regrs' in key):
        center.append(parm)  # pooling

parms = tl_corners + br_corners + center
count = 0
for key in state_dict.keys():
    if 'backbone' in key:
        continue
    state_dict_copy[key].copy_(parms[count])
    count += 1

print('Saving converted model...')
torch.save(state_dict_copy, out_file)
print('done!')
