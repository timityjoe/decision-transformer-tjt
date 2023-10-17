# from __future__ import division
# import os
# os.environ["OMP_NUM_THREADS"] = "1"
import cv2
import numpy as np
from os import path
# from statistics import mean, variance, stdev

from .model_atari import GPT

# import time
from loguru import logger
# logger.remove()
# logger.add(sys.stdout, level="INFO")
# logger.add(sys.stdout, level="SUCCESS")
# logger.add(sys.stdout, level="WARNING")

def min_max(x, mins, maxs, axis=None):
    result = (x - mins)/(maxs - mins)
    return result

# See model_atari.py GPT
def visualize_attention_head(gpt_model:GPT, mask_single_p:bool=True, mask_single_v:bool=True):
    if (mask_single_p and mask_single_v):
        mask_double = True
        logger.info(f"mask_double:{mask_double}")

    model_atts = []
    model_atts_p = []
    model_atts_v = []

    if (mask_single_p or mask_double):
        sy, sx, sc = gpt_model.visualizer.shape
        # att_p_map = np.zeros((sy, sx))
        model_att_p = gpt_model.att_p_sig5.cpu()
        model_att_p = model_att_p.numpy()
        model_att_p = model_att_p[0]
        model_att_p = model_att_p.transpose(1,2,0)[np.newaxis, :, :, :]
        model_atts_p = model_att_p if model_atts_p == [] else np.concatenate([model_atts_p,model_att_p])
    if (mask_single_v or mask_double):
        # att_v_map = np.zeros((sy, sx))
        model_att_v = gpt_model.att_v_sig5.cpu()
        model_att_v = model_att_v.numpy()
        model_att_v = model_att_v[0]
        model_att_v = model_att_v.transpose(1,2,0)[np.newaxis, :, :, :]
        model_atts_v = model_att_v if model_atts_v == [] else np.concatenate([model_atts_v,model_att_v])

    if (mask_single_p or mask_single_v or mask_double):
        # normalization (mask-attention)
        if mask_single_p or mask_double:
            max_len = model_atts_p.max(axis=None, keepdims=True)
            min_len = model_atts_p.min(axis=None, keepdims=True)
            model_atts_p = min_max(model_atts_p, min_len, max_len)
        if mask_single_v or mask_double:
            max_len = model_atts_v.max(axis=None, keepdims=True)
            min_len = model_atts_v.min(axis=None, keepdims=True)
            model_atts_v = min_max(model_atts_v, min_len, max_len)
        for i in range(img_idx):
            raw_save_path = path.join(raw_save_dir, 'raw_{0:06d}.png'.format(i))
            raw_img = cv2.imread(raw_save_path)
            #mask-attention save
            if mask_single_p or mask_double:
                att_map_p = np.zeros((sy, sx))
                model_att_p = model_atts_p[i] * 255.
                cv2.imwrite('./att.png', model_att_p)
                res1_att_p = cv2.resize(model_att_p, (80, dim))
                res2_att_p = cv2.resize(res1_att_p, (160, 160 - crop1 + crop2))
                att_map_p[crop1 : crop2 + 160, : 160] = res2_att_p
                att_map_p = cv2.applyColorMap(att_map_p.astype(np.uint8), cv2.COLORMAP_JET)
                att_map_p = cv2.addWeighted(raw_img, 0.7, att_map_p, 0.3, 0)
                #att_map_p = cv2.addWeighted(raw_img, 1.0, att_map_p, 1.0, 0)
                att_p_save_path = path.join(att_p_save_dir, 'att_p_{0:06d}.png'.format(i))
                cv2.imwrite(att_p_save_path, att_map_p)
            if mask_single_v or mask_double:
                att_map_v = np.zeros((sy, sx))
                model_att_v = model_atts_v[i] * 255.
                cv2.imwrite('./att.png', model_att_v)
                res1_att_v = cv2.resize(model_att_v, (80, dim))
                res2_att_v = cv2.resize(res1_att_v, (160, 160 - crop1 + crop2))
                att_map_v[crop1 : crop2 + 160, : 160] = res2_att_v
                att_map_v = cv2.applyColorMap(att_map_v.astype(np.uint8), cv2.COLORMAP_JET)
                att_map_v = cv2.addWeighted(raw_img, 0.7, att_map_v, 0.3, 0)
                #att_map_v = cv2.addWeighted(raw_img, 1.0, att_map_v, 1.0, 0)
                att_v_save_path = path.join(att_v_save_dir, 'att_v_{0:06d}.png'.format(i))
                cv2.imwrite(att_v_save_path, att_map_v)
