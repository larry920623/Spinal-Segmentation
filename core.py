#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
core.py - 核心推論與後處理模組
封裝模型載入、前處理、推論及所有後處理演算法，提供給 Celery Worker 呼叫。
"""

import os
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

# ---------------- CONFIG ----------------
# ⚠️ 注意：這裡使用相對路徑。請確保執行時，同目錄下有 output3/unet_best_fold3.pth
MODEL_PATH   = os.getenv("MODEL_PATH", "output3/unet_best_current_run.pth")
ENCODER_NAME = "resnet34"
TARGET_W, TARGET_H = 224, 512

# 強制使用 CPU (配合無 GPU 環境與 Docker 部署)
DEVICE = torch.device("cpu")

# Postprocess / split parameters
BONE_THR = 0.45
BND_THR  = 0.40
DT_REL   = 0.35
MIN_AREA = 400
MIN_DIST = 18
TALL_SPLIT_FACTOR = 3
MAX_SPLIT_PER_BOX = 6

# --- size constraints / splitting (max-area / height) ---
MAX_AREA = 4200
MAX_HEIGHT = 95
ROW_BND_THR = 0.42
MIN_GAP_ROWS = 22
DROP_TOO_BIG_AFTER_SPLIT = 9000

# --- NEW: anti-overlap & missing-fill parameters ---
GAP_FACTOR_FOR_MISSING = 1.60
NEWBOX_MAX_PER_GAP     = 3
BOX_H_MIN_RATIO        = 0.55
CLAMP_W_RANGE          = (26, 80)
CLAMP_H_RANGE          = (20, 90)

# ---------------- Model ----------------
class UNetWithCountBoundary(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights=None, in_channels=1, out_classes=2):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,   # bone+boundary
            activation=None,
        )
        enc_ch = self.unet.encoder.out_channels[-1]
        self.count_head = nn.Sequential(
            nn.Conv2d(enc_ch, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.count_fc = nn.Linear(64, 1)

    def forward(self, x):
        feats = self.unet.encoder(x)
        deepest = feats[-1]
        dec = self.unet.decoder(*feats)
        logits2 = self.unet.segmentation_head(dec)  # (B,2,H,W)
        c = self.count_head(deepest)
        c = torch.flatten(c, 1)
        count_pred = self.count_fc(c)
        return logits2, count_pred


def load_trained_model(weight_path, device=DEVICE, encoder_name=ENCODER_NAME):
    model = UNetWithCountBoundary(encoder_name=encoder_name, encoder_weights=None,
                                  in_channels=1, out_classes=2).to(device)
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Model weights not found at {weight_path}")
        
    state = torch.load(weight_path, map_location=device)
    if "state_dict" in state:
        state = state["state_dict"]
    try:
        model.load_state_dict(state, strict=True)
    except Exception:
        new_state = {k.replace("unet.", "").replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(new_state, strict=False)
    model.eval()
    return model

# 啟動時即將模型載入記憶體
print("Loading model to CPU...")
MODEL = load_trained_model(MODEL_PATH, DEVICE, ENCODER_NAME)
print("Model loaded successfully.")

# ---------------- Preprocessing ----------------
def preprocess_image(img_pil):
    img_gray = np.array(img_pil.convert("L"))
    img_gray = cv2.resize(img_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    img_f = (img_gray.astype(np.float32) / 255.0)[None, None, :, :]
    return torch.tensor(img_f, dtype=torch.float32), img_gray


# --------- Watershed instance split using boundary channel ----------
def instance_split_with_boundary(bone_prob, bnd_prob):
    H, W = bone_prob.shape
    fg = (bone_prob >= BONE_THR).astype(np.uint8)
    barrier = (bnd_prob >= BND_THR).astype(np.uint8)
    fg_barred = cv2.bitwise_and(fg, cv2.bitwise_not(barrier))

    dt = cv2.distanceTransform((fg_barred>0).astype(np.uint8), cv2.DIST_L2, 5)
    dt = (dt / (dt.max() + 1e-6)).astype(np.float32)
    seeds = (dt >= DT_REL).astype(np.uint8)
    seeds = cv2.morphologyEx(seeds, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    num_markers, markers = cv2.connectedComponents(seeds)

    if num_markers <= 1:
        single = np.zeros((H,W), np.int32)
        single[fg_barred>0] = 1
        return single

    energy = (1.0 - dt) * 255.0
    energy = energy.astype(np.uint8)
    energy_rgb = cv2.cvtColor(energy, cv2.COLOR_GRAY2BGR)
    markers_ws = markers.copy().astype(np.int32)
    markers_ws[(fg_barred==0)] = 0
    cv2.watershed(energy_rgb, markers_ws)
    markers_ws[markers_ws<0] = 0
    return markers_ws


def boxes_and_centroids_from_labels(labels, min_area=MIN_AREA):
    boxes, cents = [], []
    K = labels.max()
    for k in range(1, K+1):
        comp = (labels==k).astype(np.uint8)
        area = int(comp.sum())
        if area < min_area:
            continue
        ys, xs = np.where(comp>0)
        y1, y2 = int(ys.min()), int(ys.max())
        x1, x2 = int(xs.min()), int(xs.max())
        boxes.append((x1,y1,x2,y2))
        cents.append((int(xs.mean()+0.5), int(ys.mean()+0.5)))
    order = np.argsort([c[1] for c in cents])
    boxes = [boxes[i] for i in order]
    cents = [cents[i] for i in order]
    return boxes, cents


# --- 大框切割工具（沿 boundary 橫向 profile） ---
def _pick_cut_rows(row_profile, thr=0.42, min_gap=20):
    idxs = np.where(row_profile >= thr)[0]
    if len(idxs) == 0:
        return []
    cuts = []
    start = idxs[0]; prev = idxs[0]
    for i in idxs[1:]:
        if i - prev > 1:
            seg = np.arange(start, prev+1)
            cuts.append(int(seg.mean()+0.5))
            start = i
        prev = i
    seg = np.arange(start, prev+1)
    cuts.append(int(seg.mean()+0.5))
    final = []
    for c in cuts:
        if not final or (c - final[-1]) >= min_gap:
            final.append(c)
    return final

def split_large_components_by_boundary(bboxes, bnd_prob):
    out = []
    H, W = bnd_prob.shape
    for (x1,y1,x2,y2) in bboxes:
        w, h = x2-x1, y2-y1
        area = w * h
        if area <= MAX_AREA and h <= MAX_HEIGHT:
            out.append((x1,y1,x2,y2))
            continue
        yy1, yy2 = max(0,y1), min(H,y2)
        xx1, xx2 = max(0,x1), min(W,x2)
        sub = bnd_prob[yy1:yy2, xx1:xx2]
        if sub.size == 0:
            out.append((x1,y1,x2,y2)); continue
        row_profile = sub.mean(axis=1)
        cuts_rel = _pick_cut_rows(row_profile, thr=ROW_BND_THR, min_gap=MIN_GAP_ROWS)
        if len(cuts_rel) == 0:
            out.append((x1,y1,x2,y2)); continue
        cuts = [y1 + c for c in cuts_rel if (y1+c) > y1+4 and (y1+c) < y2-4]
        if len(cuts) == 0:
            out.append((x1,y1,x2,y2)); continue
        ys = [y1] + cuts + [y2]
        for i in range(len(ys)-1):
            yy_a, yy_b = int(ys[i]), int(ys[i+1])
            if yy_b - yy_a >= 8:
                out.append((x1, yy_a, x2, yy_b))
    cleaned = []
    for (x1,y1,x2,y2) in out:
        if (x2-x1)*(y2-y1) > DROP_TOO_BIG_AFTER_SPLIT:
            continue
        cleaned.append((x1,y1,x2,y2))
    cleaned.sort(key=lambda b: b[1])
    return cleaned


def vertical_split_tall_boxes(boxes):
    if not boxes:
        return boxes
    hs = [y2-y1 for (x1,y1,x2,y2) in boxes]
    med_h = np.median(hs)
    out = []
    for (x1,y1,x2,y2) in boxes:
        h = y2-y1
        if med_h > 0 and h > TALL_SPLIT_FACTOR * med_h:
            n = int(round(h / med_h))
            n = max(2, min(n, MAX_SPLIT_PER_BOX))
            step = h / n
            for i in range(n):
                yy1 = int(y1 + i*step)
                yy2 = int(y1 + (i+1)*step)
                out.append((x1, yy1, x2, yy2))
        else:
            out.append((x1,y1,x2,y2))
    out = sorted(out, key=lambda b: b[1])
    return out


# ================= NEW: anti-overlap & missing-fill =================
def _median_dims_and_centerline(boxes, W, H):
    ws = [x2-x1 for (x1,y1,x2,y2) in boxes]
    hs = [y2-y1 for (x1,y1,x2,y2) in boxes]
    csx = [ (x1+x2)/2 for (x1,y1,x2,y2) in boxes ]
    med_w = float(np.median(ws)) if ws else 30.0
    med_h = float(np.median(hs)) if hs else 40.0
    med_w = float(np.clip(med_w, CLAMP_W_RANGE[0], CLAMP_W_RANGE[1]))
    med_h = float(np.clip(med_h, CLAMP_H_RANGE[0], CLAMP_H_RANGE[1]))
    x_med = int(np.clip(np.median(csx) if csx else W/2, 0, W-1))
    return med_w, med_h, x_med

def enforce_non_overlap_sorted(boxes, med_h, H):
    if len(boxes) <= 1:
        return boxes
    cents_y = [ (b[1]+b[3])//2 for b in boxes ]
    cuts = [ (cents_y[i] + cents_y[i+1])//2 for i in range(len(cents_y)-1) ]
    out = list(boxes)
    for i in range(len(out)-1):
        x1,y1,x2,y2 = out[i]
        nx1,ny1,nx2,ny2 = out[i+1]
        mid = cuts[i]
        y2_new = min(y2, mid)
        ny1_new = max(ny1, mid)
        min_h = int(max(8, med_h*BOX_H_MIN_RATIO))
        if (y2_new - y1) < min_h:
            cy = (y1 + y2_new)//2
            y1 = max(0, cy - min_h//2)
            y2_new = min(H, y1 + min_h)
        if (ny2 - ny1_new) < min_h:
            cy = (ny1_new + ny2)//2
            ny1_new = max(0, cy - min_h//2)
            ny2 = min(H, ny1_new + min_h)
        out[i]   = (x1, y1, x2, y2_new)
        out[i+1] = (nx1, ny1_new, nx2, ny2)
    return out

def fill_missing_boxes_sorted(boxes, med_w, med_h, x_med, H):
    if len(boxes) <= 1:
        return boxes
    cents = [ ((b[0]+b[2])//2, (b[1]+b[3])//2) for b in boxes ]
    gaps = [ cents[i+1][1] - cents[i][1] for i in range(len(cents)-1) ]
    step_med = float(np.median(gaps)) if gaps else med_h
    if not np.isfinite(step_med) or step_med <= 0:
        step_med = med_h
    out = list(boxes)
    added = []
    for i, gap in enumerate(gaps):
        if gap <= GAP_FACTOR_FOR_MISSING * step_med:
            continue
        n_new = int(np.clip(round(gap / step_med) - 1, 1, NEWBOX_MAX_PER_GAP))
        y_start = cents[i][1]
        step = gap / (n_new + 1)
        for k in range(n_new):
            cy = int(round(y_start + (k+1)*step))
            h = int(round(med_h))
            w = int(round(med_w))
            y1 = int(np.clip(cy - h//2, 0, H-1))
            y2 = int(np.clip(y1 + h, 0, H))
            x1 = int(np.clip(x_med - w//2, 0, TARGET_W-1))
            x2 = int(np.clip(x1 + w, 0, TARGET_W))
            added.append((x1,y1,x2,y2))
    if added:
        out = sorted(out + added, key=lambda b: (b[1]+b[3])//2)
    return out

def harmonize_and_fix_boxes(boxes, imgW, imgH):
    if not boxes:
        return boxes
    med_w, med_h, x_med = _median_dims_and_centerline(boxes, imgW, imgH)
    aligned = sorted(boxes, key=lambda b: (b[1] + b[3]) // 2)
    filled = fill_missing_boxes_sorted(aligned, med_w, med_h, x_med, imgH)
    nonoverlap = enforce_non_overlap_sorted(filled, med_h, imgH)
    
    final = []
    min_h = int(max(8, med_h * BOX_H_MIN_RATIO))
    for (x1, y1, x2, y2) in nonoverlap:
        h = y2 - y1
        if h < min_h:
            cy = (y1 + y2) // 2
            y1 = max(0, cy - min_h // 2)
            y2 = min(imgH, y1 + min_h)
        final.append((x1, y1, x2, y2))

    final.sort(key=lambda b: b[1])
    return final


# ---------------- Inference Pipeline ----------------
def infer_and_extract(img_tensor):
    """
    執行推論並回傳結果。
    為了相容 JSON 序列化 (Celery)，所有的 np.int / np.float 皆轉換為標準 python type。
    """
    img_tensor = img_tensor.to(DEVICE)
    with torch.no_grad():
        logits2, count_pred = MODEL(img_tensor)
        bone_log = logits2[:,0:1]
        bnd_log  = logits2[:,1:2]

        if bone_log.shape[2:] != (TARGET_H, TARGET_W):
            bone_log = F.interpolate(bone_log, size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)
            bnd_log  = F.interpolate(bnd_log,  size=(TARGET_H, TARGET_W), mode="bilinear", align_corners=False)

        bone_prob = torch.sigmoid(bone_log)[0,0].cpu().numpy().astype(np.float32)
        bnd_prob  = torch.sigmoid(bnd_log)[0,0].cpu().numpy().astype(np.float32)
        count_val = float(count_pred.view(-1)[0].cpu().item())

    labels = instance_split_with_boundary(bone_prob, bnd_prob)
    boxes, cents = boxes_and_centroids_from_labels(labels, min_area=MIN_AREA)

    boxes = split_large_components_by_boundary(boxes, bnd_prob)
    boxes = vertical_split_tall_boxes(boxes)
    boxes = harmonize_and_fix_boxes(boxes, TARGET_W, TARGET_H)

    # 確保回傳值為標準 Python 內建型別，避免 JSON 序列化失敗
    clean_boxes = [tuple(int(x) for x in b) for b in boxes]
    clean_cents = [tuple(int(x) for x in c) for c in cents]

    return bone_prob, bnd_prob, clean_boxes, clean_cents, float(count_val)

def dice_np(pred, mask, eps=1e-6):
    pred_b = (pred > 0).astype(np.uint8)
    mask_b = (mask > 0).astype(np.uint8)
    inter = int((pred_b & mask_b).sum())
    denom = int(pred_b.sum() + mask_b.sum())
    return float((2*inter + eps) / (denom + eps))

def calculate_metrics(bone_prob, gt_pil):
    # 將 GT 轉為與模型輸出一致的尺寸與格式
    gt_gray = np.array(gt_pil.convert("L"))
    gt_resized = cv2.resize(gt_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
    gt_mask = (gt_resized > 127).astype(np.uint8) * 255
    
    # 預測遮罩 (使用預設門檻)
    pred_mask = (bone_prob >= BONE_THR).astype(np.uint8) * 255
    dice_val = dice_np(pred_mask, gt_mask)
    
    # 計算 GT 實體數量
    num_labels, _ = cv2.connectedComponents((gt_mask > 0).astype(np.uint8), connectivity=8)[:2]
    gt_count = max(0, num_labels - 1)
    
    return dice_val, gt_count