#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
a.py  (with boundary aux head + count-align loss)
- Dataset, training (BCE/Dice for bone+boundary), count head, KFold, accumulation, TTA, ensemble,
  threshold search, postprocess, plots.
- Adds:
  (1) Boundary auxiliary channel supervision
  (2) Small-weight count-align loss between segmentation mass and predicted count
"""

import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import KFold
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- segmentation_models_pytorch ----
try:
    import segmentation_models_pytorch as smp
except Exception as e:
    raise ImportError("需要安裝 segmentation-models-pytorch 與 timm：\n pip install segmentation-models-pytorch timm\n錯誤訊息：" + str(e))

# ----------------- CONFIG -----------------
ORIG_TARGET_W = 214
ORIG_TARGET_H = 512

def make_divisible(x, d=32):
    return ((x + d - 1) // d) * d

TARGET_W = make_divisible(ORIG_TARGET_W, 32)   # -> 224
TARGET_H = make_divisible(ORIG_TARGET_H, 32)   # -> 512
if TARGET_W != ORIG_TARGET_W or TARGET_H != ORIG_TARGET_H:
    print(f"[Info] adjusted target size from ({ORIG_TARGET_W},{ORIG_TARGET_H}) to ({TARGET_W},{TARGET_H}) to be divisible by 32.")

# batch / training
BATCH_SIZE = 2
EFFECTIVE_BATCH_SIZE = 4
LR = 1e-3
EPOCHS = 100
PATIENCE = 10
FINETUNE = True
FINETUNE_LR_FACTOR = 0.1
FINETUNE_EPOCHS = 30
FINETUNE_PATIENCE = 8

# *** 輸出路徑改為 output3 ***
OUTPUT_DIR = "/home/sivslab/文件/新生訓練/AI新生訓練/output_SMP"
os.makedirs(OUTPUT_DIR, exist_ok=True)
DEBUG_SAVE_DIR = os.path.join(OUTPUT_DIR, "debug_samples"); os.makedirs(DEBUG_SAVE_DIR, exist_ok=True)
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots"); os.makedirs(PLOTS_DIR, exist_ok=True)
ENSEMBLE_DIR = os.path.join(OUTPUT_DIR, "ensemble_results"); os.makedirs(ENSEMBLE_DIR, exist_ok=True)

NUM_WORKERS = 0
EPS = 1e-6
USE_AUG = False
SEED = 42
REDUCE_LR_ON_PLATEAU = True
VERBOSE_DEBUG_STATS = True
POS_WEIGHT_CAP = 50.0

# UNet / encoder config
ENCODER_NAME = 'resnet34'
ENCODER_WEIGHTS = 'imagenet'
IN_CHANNELS = 1

# heads / losses weights
LAMBDA_DICE          = 1.0   # bone 通道 Dice 權重
LAMBDA_BCE_MAIN      = 1.0   # bone 通道 BCE 權重
LAMBDA_BOUNDARY      = 1.0   # boundary 通道 (BCE+Dice) 整體權重
LAMBDA_COUNT         = 2.0   # head 回歸 count 權重
LAMBDA_COUNT_ALIGN   = 1.0   # seg 連通元件數對齊 GT 權重

USE_DICE = True

# ensemble / post-process
ENSEMBLE_TOPK = 2
TTA_ENABLED = True
POST_MIN_AREA = 50

# accumulation
if EFFECTIVE_BATCH_SIZE % BATCH_SIZE != 0:
    raise ValueError("EFFECTIVE_BATCH_SIZE must be multiple of BATCH_SIZE")
ACCUM_STEPS = EFFECTIVE_BATCH_SIZE // BATCH_SIZE
print(f"Config: TARGET {TARGET_W}x{TARGET_H}, BATCH_SIZE={BATCH_SIZE}, EFFECTIVE_BATCH_SIZE={EFFECTIVE_BATCH_SIZE}, ACCUM_STEPS={ACCUM_STEPS}")

torch.manual_seed(SEED); np.random.seed(SEED); random.seed(SEED)

# ------------ Utilities ------------
def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_debug_image(inp_arr, gt_arr, pred_arr, save_path):
    H, W = inp_arr.shape
    inp_u  = (np.clip(inp_arr, 0, 1) * 255).astype(np.uint8)
    gt_u   = (np.clip(gt_arr, 0, 1) * 255).astype(np.uint8)
    pred_u = (np.clip(pred_arr, 0, 1) * 255).astype(np.uint8)
    panel_w = W * 3 + 40
    canvas = Image.new("RGB", (panel_w, H + 30), color=(30, 30, 30))
    pil_inp  = Image.fromarray(inp_u).convert("RGB")
    pil_gt   = Image.fromarray(gt_u).convert("RGB")
    pil_pred = Image.fromarray(pred_u).convert("RGB")
    canvas.paste(pil_inp, (0, 0))
    canvas.paste(pil_gt, (W + 20, 0))
    canvas.paste(pil_pred, (2 * (W + 20), 0))
    draw = ImageDraw.Draw(canvas)
    try: fnt = ImageFont.load_default()
    except: fnt = None
    draw.text((4, H + 4), "Input (resized)", fill=(255, 255, 255), font=fnt)
    draw.text((W + 24, H + 4), "GT (bone)", fill=(255, 255, 255), font=fnt)
    draw.text((2 * (W + 20) + 4, H + 4), "Pred bone prob", fill=(255, 255, 255), font=fnt)
    canvas.save(save_path)

def make_boundary_from_mask(mask_u8, k=3):
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dil = cv2.dilate(mask_u8, kernel, iterations=1)
    ero = cv2.erode(mask_u8,  kernel, iterations=1)
    bd = cv2.subtract(dil, ero)
    bd = (bd > 0).astype(np.uint8)
    return bd

# ---------- Dataset ----------
class JsonSpineDataset(Dataset):
    def __init__(self, img_json_files, label_json_files, img_roots, label_roots,
                 target_w=TARGET_W, target_h=TARGET_H, use_aug=USE_AUG):
        self.img_roots = img_roots
        self.label_roots = label_roots
        self.target_w = target_w
        self.target_h = target_h
        self.use_aug = use_aug

        img_dict = {}
        for jf in img_json_files:
            if not os.path.exists(jf):
                print(f"[Warning] image json not found: {jf}"); continue
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get('images', []):
                    fname = item.get('file_name')
                    if fname: img_dict[fname] = fname

        label_dict = {}
        for jf in label_json_files:
            if not os.path.exists(jf):
                print(f"[Warning] label json not found: {jf}"); continue
            with open(jf, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get('images', []):
                    fname = item.get('file_name')
                    if fname: label_dict[fname] = fname

        candidates = sorted(list(set(img_dict.keys()) & set(label_dict.keys())))
        if len(candidates) == 0:
            raise ValueError("No matching filenames between image JSONs and label JSONs.")

        valid_filenames, missing_img, missing_label = [], [], []
        for fname in candidates:
            found_img   = any(os.path.exists(os.path.join(root, fname)) for root in self.img_roots)
            found_label = any(os.path.exists(os.path.join(root, fname)) for root in self.label_roots)
            if found_img and found_label:
                valid_filenames.append(fname)
            else:
                if not found_img:   missing_img.append(fname)
                if not found_label: missing_label.append(fname)

        self.filenames = sorted(valid_filenames)
        print(f"[Dataset] total candidates (JSON intersection): {len(candidates)}")
        print(f"[Dataset] usable samples (both exist): {len(self.filenames)}")
        if len(missing_img)>0:   print(f"[Dataset] missing image files: {len(missing_img)} (examples: {missing_img[:5]})")
        if len(missing_label)>0: print(f"[Dataset] missing label files: {len(missing_label)} (examples: {missing_label[:5]})")
        if len(self.filenames) == 0:
            raise ValueError("No usable image+label pairs found on disk.")

    def __len__(self): return len(self.filenames)

    def _random_augment(self, img_pil, mask_pil):
        if random.random() < 0.5:
            img_pil  = img_pil.transpose(Image.FLIP_LEFT_RIGHT)
            mask_pil = mask_pil.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            angle = random.uniform(-10, 10)
            img_pil  = img_pil.rotate(angle,  resample=Image.BILINEAR, fillcolor=0)
            mask_pil = mask_pil.rotate(angle, resample=Image.NEAREST,  fillcolor=0)
        return img_pil, mask_pil

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img_path, label_path = None, None
        for root in self.img_roots:
            p = os.path.join(root, fname)
            if os.path.exists(p): img_path = p; break
        for root in self.label_roots:
            p = os.path.join(root, fname)
            if os.path.exists(p): label_path = p; break
        if img_path is None or label_path is None:
            raise FileNotFoundError(f"{fname} missing on disk (img: {img_path}, label: {label_path})")

        img_pil   = Image.open(img_path).convert('L')
        label_pil = Image.open(label_path).convert('L')
        if self.use_aug:
            img_pil, label_pil = self._random_augment(img_pil, label_pil)

        img_np   = cv2.resize(np.array(img_pil),   (self.target_w, self.target_h), interpolation=cv2.INTER_AREA)
        label_np = cv2.resize(np.array(label_pil), (self.target_w, self.target_h), interpolation=cv2.INTER_NEAREST)

        if label_np.max()>1: label_np = (label_np>127).astype(np.uint8)
        bone_u8 = label_np.astype(np.uint8)
        boundary_u8 = make_boundary_from_mask(bone_u8, k=3)

        # gt_count by connected components
        num_labels, _, _, _ = cv2.connectedComponentsWithStats(bone_u8, connectivity=8)
        gt_count = max(0, num_labels-1)

        img_f   = (img_np.astype(np.float32)/255.0)[None, ...]  # (1,H,W)
        bone_f  = bone_u8.astype(np.float32)[None, ...]          # (1,H,W)
        bnd_f   = boundary_u8.astype(np.float32)[None, ...]      # (1,H,W)
        y2 = np.concatenate([bone_f, bnd_f], axis=0)             # (2,H,W)

        return (
            torch.tensor(img_f, dtype=torch.float32),               # x
            torch.tensor(y2,   dtype=torch.float32),                # y: 2ch
            torch.tensor([gt_count], dtype=torch.float32),          # gt_count
            fname
        )

# ---------- Model ----------
class UNetWithCountBoundary(nn.Module):
    def __init__(self, encoder_name='resnet34', encoder_weights='imagenet', in_channels=1, out_classes=2):
        super().__init__()
        self.unet = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=out_classes,
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
        # 正確做法：直接用 encoder(x) 取得 features list
        features = self.unet.encoder(x)         # list: [x0, x1, x2, x3, x4]
        deepest  = features[-1]                 # 最深層 feature (BxC5xH/32xW/32)

        # decoder 需要的是 *features[::-1] 逐個解包，而不是把 list 當單一參數
        dec = self.unet.decoder(*features)  # <-- 重點：使用解包 *
        logits_2 = self.unet.segmentation_head(dec)  # (B,2,H,W)

        # count head 用最深層特徵
        c = self.count_head(deepest)             # Bx64x1x1
        c = torch.flatten(c, 1)                  # Bx64
        count_pred = self.count_fc(c)            # Bx1

        return logits_2, count_pred

# ---------- Losses & Metrics ----------
def dice_loss_from_logits_binary(logits, target, eps=EPS):
    """ logits,target shape: (B,1,H,W) """
    probs = torch.sigmoid(logits)
    num = 2.0 * (probs * target).sum(dim=(1,2,3)) + eps
    den = probs.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3)) + eps
    dice_per = num / den
    return (1.0 - dice_per).mean()

def dice_metric_from_logits_binary(logits, target, eps=EPS, thr=0.5):
    probs = torch.sigmoid(logits)
    pred_bin = (probs > thr).float()
    inter = (pred_bin * target).sum(dim=(1,2,3))
    denom = pred_bin.sum(dim=(1,2,3)) + target.sum(dim=(1,2,3))
    dice_per = (2*inter + eps) / (denom + eps)
    return float(dice_per.mean().item())

def compute_pos_weight(dataset):
    pos, total = 0, 0
    for i in range(len(dataset)):
        _, y2, _, _ = dataset[i]
        mask_np = y2.numpy()[0]   # bone channel
        pos   += int((mask_np > 0.5).sum())
        total += mask_np.size
    neg = total - pos
    if pos == 0:
        print("[Warning] no positive pixels found; pos_weight=1.0")
        return 1.0
    pw = float(neg)/float(pos)
    pw = max(1.0, min(pw, POS_WEIGHT_CAP))
    print(f"[pos_weight] pos={pos}, neg={neg}, pos_weight={pw:.3f}")
    return pw

def dice_np(pred, mask, eps=1e-6):
    pred_b = (pred > 0).astype(np.uint8)
    mask_b = (mask > 0).astype(np.uint8)
    inter = int((pred_b & mask_b).sum())
    denom = int(pred_b.sum() + mask_b.sum())
    return float((2*inter + eps) / (denom + eps))

def find_best_threshold(probs_all, masks_all, thr_list=None):
    if probs_all.ndim == 4:
        probs = probs_all[:, 0]   # (N,H,W)
    else:
        probs = probs_all
    if masks_all.ndim == 4:
        masks = masks_all[:, 0]
    else:
        masks = masks_all
    if thr_list is None:
        thr_list = np.linspace(0.1, 0.9, 17)
    best_t, best_d = 0.5, -1.0
    for t in thr_list:
        preds = (probs > t).astype(np.uint8)
        dices = [dice_np(preds[i], masks[i]) for i in range(len(preds))]
        mean_d = float(np.mean(dices))
        if mean_d > best_d:
            best_d, best_t = mean_d, t
    return best_t, best_d

def postprocess_mask(mask_bin, min_area=POST_MIN_AREA):
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_bin.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask_bin, dtype=np.uint8)
    for lbl in range(1, num_labels):
        area = stats[lbl, cv2.CC_STAT_AREA]
        if area >= min_area:
            out[labels == lbl] = 1
    kernel = np.ones((3,3), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel, iterations=1)
    return out

def tta_predict(model, img_tensor, device):
    model.eval()
    with torch.no_grad():
        imgs = img_tensor.to(device)
        out1, _ = model(imgs)
        p1 = torch.sigmoid(out1[:,0:1]).cpu().numpy()  # bone channel
        if not TTA_ENABLED:
            return p1
        imgs_f = torch.flip(imgs, dims=[3])
        out2, _ = model(imgs_f)
        p2 = torch.sigmoid(out2[:,0:1]).cpu().numpy()
        p2 = np.flip(p2, axis=3)
        probs = (p1 + p2) / 2.0
    return probs

def evaluate_ensemble_on_dataset(model_paths, dataset, device, save_results_dir=ENSEMBLE_DIR):
    ensure_dir(save_results_dir)
    N, H, W = len(dataset), TARGET_H, TARGET_W
    probs_all = np.zeros((N, 1, H, W), dtype=np.float32)
    masks_all = np.zeros((N, 1, H, W), dtype=np.uint8)
    fnames = []
    models = []
    for mp in model_paths:
        m = UNetWithCountBoundary(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS, in_channels=IN_CHANNELS, out_classes=2).to(device)
        m.load_state_dict(torch.load(mp, map_location=device))
        m.eval()
        models.append(m)

    for i in range(N):
        x, y2, _, fname = dataset[i]
        fnames.append(fname)
        img_t = x.unsqueeze(0)
        bone_np = y2[0].numpy().astype(np.uint8)
        masks_all[i, 0] = bone_np
        prob_acc = np.zeros((1, 1, H, W), dtype=np.float32)
        for m in models:
            p = tta_predict(m, img_t, device)
            prob_acc += p
        prob_acc /= max(1, len(models))
        probs_all[i] = prob_acc[0]
    best_t, best_d = find_best_threshold(probs_all, masks_all)
    print(f"[Ensemble] best threshold={best_t:.3f}, best dice (before postprocess)={best_d:.4f}")

    per_dices = []
    for i in range(N):
        prob = probs_all[i, 0]
        pred = (prob > best_t).astype(np.uint8)
        pred_pp = postprocess_mask(pred, min_area=POST_MIN_AREA)
        dice_after = dice_np(pred_pp, masks_all[i,0])
        per_dices.append(dice_after)
        if i < 50:
            img_np = dataset[i][0].squeeze(0).numpy()
            gt_np  = masks_all[i,0]
            pred_vis = pred_pp
            outpath = os.path.join(save_results_dir, f"ens_{i:03d}_{fnames[i]}")
            Hh, Ww = img_np.shape
            canvas = Image.new("RGB", (Ww*3 + 40, Hh + 30), color=(30,30,30))
            pil_inp  = Image.fromarray((img_np*255).astype(np.uint8)).convert("RGB")
            pil_gt   = Image.fromarray((gt_np*255).astype(np.uint8)).convert("RGB")
            pil_pred = Image.fromarray((pred_vis*255).astype(np.uint8)).convert("RGB")
            canvas.paste(pil_inp, (0,0))
            canvas.paste(pil_gt,  (Ww+20,0))
            canvas.paste(pil_pred,(2*(Ww+20),0))
            draw = ImageDraw.Draw(canvas)
            try: fnt = ImageFont.load_default()
            except: fnt = None
            draw.text((4, Hh+4), "Input", fill=(255,255,255), font=fnt)
            draw.text((Ww+24, Hh+4), "GT",    fill=(255,255,255), font=fnt)
            draw.text((2*(Ww+20)+4, Hh+4), f"Ensemble Pred (post) Dice={dice_after:.4f}", fill=(255,255,255), font=fnt)
            canvas.save(outpath + ".png")
    mean_dice_after = float(np.mean(per_dices))
    print(f"[Ensemble] mean dice after postprocess (min_area={POST_MIN_AREA}) = {mean_dice_after:.4f}")
    return best_t, best_d, mean_dice_after, per_dices

# ---------- Training ----------
def train_model(
    model,
    train_loader,
    val_loader,
    optimizer,
    criterion_bce_logits,
    epochs,
    device,
    patience=PATIENCE,
    use_dice=USE_DICE,
    lambda_dice=LAMBDA_DICE,
    accum_steps=ACCUM_STEPS,
):
    """
    - dataloader: (imgs, y2[bone,boundary], counts, fnames)
    - segmentation: BCE(骨+邊界) + Dice(骨 + 邊界*LAMBDA_BOUNDARY)
    - count losses: head + seg-cc align
    - 每個 epoch 印出計數統計
    """
    def dice_loss_binary(logits1c, target1c):
        return dice_loss_from_logits_binary(logits1c, target1c, eps=EPS)

    def dice_metric_binary(logits1c, target1c):
        return dice_metric_from_logits_binary(logits1c, target1c, eps=EPS, thr=0.5)

    def _batch_seg_cc_count_from_logits(
        logits_2,                # (B,2,H,W) -> ch0: bone, ch1: boundary
        thr_bone=0.65,
        thr_boundary=0.3,
        min_area=300,
        peak_rel=0.6,           # 距離圖峰值比例門檻（控制切割顆數；大→較少種子）
        min_dist=15               # 峰值間最小距離（像素）
    ):
        """
        用 bone + boundary 做 watershed 來估計「可數出的實體骨塊數」。
        回傳：shape [B] 的 float tensor（每張圖的顆數）
        """
        with torch.no_grad():
            bone_prob = torch.sigmoid(logits_2[:, 0:1])          # (B,1,H,W)
            bnd_prob  = torch.sigmoid(logits_2[:, 1:2])          # (B,1,H,W)

            bone_np = (bone_prob.cpu().numpy()[:, 0] > thr_bone).astype(np.uint8)  # (B,H,W)
            bnd_np  = (bnd_prob.cpu().numpy()[:, 0] > thr_boundary).astype(np.uint8)

            counts = []
            for m_bin, bnd_bin in zip(bone_np, bnd_np):
                # 1) 先基本去噪 + 最小面積過濾
                m_bin = postprocess_mask(m_bin, min_area=min_area)

                # 2) 距離變換產生內部種子
                dist = cv2.distanceTransform((m_bin*255).astype(np.uint8), cv2.DIST_L2, 5)
                if dist.max() > 0:
                    dist_norm = dist / (dist.max() + 1e-6)
                else:
                    counts.append(0.0)
                    continue

                # 3) 取局部峰值作種子（簡版：閾值 + 非極大值抑制）
                peak_mask = (dist_norm >= peak_rel).astype(np.uint8)

                # 粗略的非極大值抑制：用最大濾波 + 比對
                k = max(3, min_dist | 1)  # odd
                max_f = cv2.dilate(dist_norm, np.ones((k, k), np.uint8))
                nms = ((dist_norm == max_f) & (dist_norm >= peak_rel)).astype(np.uint8)

                # 確保在骨區內
                seeds_bin = (nms > 0).astype(np.uint8)
                seeds_bin[m_bin == 0] = 0

                # 至少要有一個種子，否則以整塊當一塊
                if seeds_bin.sum() < 1 and m_bin.sum() > 0:
                    seeds_bin = np.zeros_like(m_bin, dtype=np.uint8)
                    # 取距離圖最大點做 1 個種子
                    yx = np.unravel_index(np.argmax(dist), dist.shape)
                    seeds_bin[yx] = 1

                # 4) 產生 marker（int32, 1..K）
                num_seeds, markers, _, _ = cv2.connectedComponentsWithStats(seeds_bin, connectivity=8)
                if num_seeds <= 1:
                    # 沒有或只有一個種子 ⇒ 一塊
                    cc = 1 if m_bin.sum() > 0 else 0
                    counts.append(float(cc))
                    continue

                # 5) 準備 watershed 輸入
                #   OpenCV 的 watershed 需要 3ch 影像；我們用骨區做灰階貼到3ch即可
                img3 = np.dstack([m_bin*255]*3).astype(np.uint8)

                # 邊界當作阻擋：把明顯邊界像素設為未知區域（0），避免連在一起
                unknown = cv2.dilate(bnd_bin, np.ones((3,3), np.uint8), iterations=1)
                markers[unknown > 0] = 0

                # 6) 跑 watershed（就地會把邊界標成 -1）
                markers = markers.astype(np.int32)
                cv2.watershed(img3, markers)

                # 7) 計數（>1 的 label 視為實體；-1 是邊界）
                labels = markers.copy()
                labels[labels <= 1] = 0
                labels[m_bin == 0]  = 0

                # 很小的切割碎片再過濾
                lbl_ids = np.unique(labels)
                cc = 0
                for lid in lbl_ids:
                    if lid <= 1: continue
                    area = int((labels == lid).sum())
                    if area >= min_area:
                        cc += 1

                counts.append(float(cc))

            return torch.tensor(counts, dtype=torch.float32, device=logits_2.device)


    best_val_dice = 0.0
    patience_counter = 0
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    history = {'train_loss': [], 'train_dice': [], 'val_loss': [], 'val_dice': []}

    for epoch in range(epochs):
        # ---------------- Train ----------------
        model.train()
        t_loss = 0.0
        t_dice = 0.0
        it = 0
        optimizer.zero_grad()

        for step, (imgs, y2, counts, fnames) in enumerate(train_loader):
            imgs   = imgs.to(device)
            y2     = y2.to(device)             # (B,2,H,W) [bone,boundary]
            counts = counts.to(device).view(-1)

            logits2, count_pred = model(imgs)  # logits2: (B,2,H,W)
            count_pred = count_pred.view(-1)

            if logits2.shape[2:] != y2.shape[2:]:
                logits2 = F.interpolate(logits2, size=y2.shape[2:], mode='bilinear', align_corners=False)

            bone_log = logits2[:,0:1]
            bnd_log  = logits2[:,1:2]
            bone_gt  = y2[:,0:1]
            bnd_gt   = y2[:,1:2]

            # BCE
            loss_bce_bone = criterion_bce_logits(bone_log, bone_gt) * LAMBDA_BCE_MAIN
            loss_bce_bnd  = criterion_bce_logits(bnd_log,  bnd_gt)  * LAMBDA_BOUNDARY

            loss = loss_bce_bone + loss_bce_bnd

            # Dice
            if use_dice:
                loss_dice_bone = dice_loss_binary(bone_log, bone_gt) * lambda_dice
                loss_dice_bnd  = dice_loss_binary(bnd_log,  bnd_gt)  * LAMBDA_BOUNDARY
                loss += (loss_dice_bone + loss_dice_bnd)

            # Count head loss
            loss_count_head = F.smooth_l1_loss(count_pred, counts) * LAMBDA_COUNT
            loss += loss_count_head

            # Seg-CC align
            seg_counts = _batch_seg_cc_count_from_logits(
                logits2,
                thr_bone=0.65,
                thr_boundary=0.3,
                min_area=300,
                peak_rel=0.6,
                min_dist=15
            )
            loss_count_align = F.smooth_l1_loss(seg_counts, counts) * LAMBDA_COUNT_ALIGN
            loss += loss_count_align

            # backward (gradient accumulation)
            (loss / accum_steps).backward()
            if (step + 1) % accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                optimizer.zero_grad()

            # 訓練 dice 只看 bone 通道
            t_loss += loss.item()
            t_dice += dice_metric_binary(bone_log.detach().cpu(), bone_gt.detach().cpu())
            it += 1

        t_loss = t_loss / max(1, it)
        t_dice = t_dice / max(1, it)

        # ---------------- Val ----------------
        model.eval()
        v_loss = 0.0
        v_dice = 0.0
        vit = 0

        gt_counts_all, head_counts_all, segcc_counts_all = [], [], []
        pred_mean_list, pred_min_list, pred_max_list, gt_pos_list = [], [], [], []

        with torch.no_grad():
            for imgs, y2, counts, fnames in val_loader:
                imgs   = imgs.to(device)
                y2     = y2.to(device)
                counts = counts.to(device).view(-1)

                logits2, count_pred = model(imgs)
                count_pred = count_pred.view(-1)

                if logits2.shape[2:] != y2.shape[2:]:
                    logits2 = F.interpolate(logits2, size=y2.shape[2:], mode='bilinear', align_corners=False)

                bone_log = logits2[:,0:1]
                bnd_log  = logits2[:,1:2]
                bone_gt  = y2[:,0:1]
                bnd_gt   = y2[:,1:2]

                loss_bce_bone = criterion_bce_logits(bone_log, bone_gt) * LAMBDA_BCE_MAIN
                loss_bce_bnd  = F.binary_cross_entropy_with_logits(bnd_log, bnd_gt) * LAMBDA_BOUNDARY
                loss = loss_bce_bone + loss_bce_bnd

                if use_dice:
                    loss_dice_bone = dice_loss_binary(bone_log, bone_gt) * lambda_dice
                    loss_dice_bnd  = dice_loss_binary(bnd_log,  bnd_gt)  * LAMBDA_BOUNDARY
                    loss += (loss_dice_bone + loss_dice_bnd)

                loss_count_head = F.smooth_l1_loss(count_pred, counts) * LAMBDA_COUNT
                loss += loss_count_head

                seg_counts = _batch_seg_cc_count_from_logits(
                    logits2,
                    thr_bone=0.65,
                    thr_boundary=0.3,
                    min_area=300,
                    peak_rel=0.6,
                    min_dist=15
                )
                loss_count_align = F.smooth_l1_loss(seg_counts, counts) * LAMBDA_COUNT_ALIGN
                loss += loss_count_align

                v_loss += loss.item()
                v_dice += dice_metric_binary(bone_log.cpu(), bone_gt.cpu())
                vit += 1

                probs_bone = torch.sigmoid(bone_log)
                pred_mean_list.append(float(probs_bone.mean().cpu().item()))
                pred_min_list.append(float(probs_bone.min().cpu().item()))
                pred_max_list.append(float(probs_bone.max().cpu().item()))
                gt_pos_list.append(float((bone_gt > 0.5).float().mean().cpu().item()))

                gt_counts_all.append(counts.detach().cpu())
                head_counts_all.append(count_pred.detach().cpu())
                segcc_counts_all.append(seg_counts.detach().cpu())

        v_loss = v_loss / max(1, vit)
        v_dice = v_dice / max(1, vit)

        gt_mean   = torch.cat(gt_counts_all).float().mean().item() if gt_counts_all else 0.0
        head_mean = torch.cat(head_counts_all).float().mean().item() if head_counts_all else 0.0
        segcc_mean= torch.cat(segcc_counts_all).float().mean().item() if segcc_counts_all else 0.0

        history['train_loss'].append(float(t_loss))
        history['train_dice'].append(float(t_dice))
        history['val_loss'].append(float(v_loss))
        history['val_dice'].append(float(v_dice))

        print(f"Epoch {epoch+1}/{epochs}  TrainLoss={t_loss:.4f}  TrainDice={t_dice:.4f}  ValLoss={v_loss:.4f}  ValDice={v_dice:.4f}")
        print(f"  [Val   Count] GT_mean={gt_mean:.3f} | Head_mean={head_mean:.3f} | SegCC_mean={segcc_mean:.3f}")

        scheduler.step(v_loss)
        if VERBOSE_DEBUG_STATS:
            lr_now = optimizer.param_groups[0]['lr']
            pm = np.mean(pred_mean_list) if pred_mean_list else 0.0
            pmin = np.min(pred_min_list) if pred_min_list else 0.0
            pmax = np.max(pred_max_list) if pred_max_list else 0.0
            gm = np.mean(gt_pos_list) if gt_pos_list else 0.0
            print(f"  [DebugStats] pred_mean={pm:.6f}, pred_min={pmin:.6f}, pred_max={pmax:.6f}, gt_pos_ratio={gm:.6f}")
            print(f"  [LR] current LR={lr_now:.6e}")

        if v_dice > best_val_dice + 1e-6:
            best_val_dice = v_dice
            patience_counter = 0
            cur_best_path = os.path.join(OUTPUT_DIR, "unet_best_current_run.pth")
            torch.save(model.state_dict(), cur_best_path)
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping (patience={patience}).")
                break

    return model, best_val_dice, history

# fine-tune helper
def finetune_from_checkpoint(model, train_loader, val_loader, checkpoint_path, device, base_lr,
                             factor=FINETUNE_LR_FACTOR, epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
                             pos_weight_bone=None):
    print(f"Finetune: loading checkpoint {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state)
    new_lr = base_lr * factor
    print(f"Finetune: setting LR to {new_lr}")
    optimizer = torch.optim.Adam(model.parameters(), lr=new_lr)
    if pos_weight_bone is None:
        criterion_bce_logits = torch.nn.BCEWithLogitsLoss()
    else:
        criterion_bce_logits = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_bone)
    model, best, history = train_model(model, train_loader, val_loader, optimizer, criterion_bce_logits,
                                       epochs=epochs, device=device, patience=patience)
    return model, best, history

# ---------- Main ----------
if __name__ == "__main__":
    # paths - adjust if needed
    img_json_files = [
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f01/f01.json",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f02/f02.json",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f03/f03.json"
    ]
    label_json_files = [
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f01/f1_gt.json",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f02/f2_gt.json",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f03/f3_gt.json"
    ]
    img_roots = [
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f01/image",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f02/image",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f03/image"
    ]
    label_roots = [
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f01/label",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f02/label",
        "/home/sivslab/文件/新生訓練/AI新生訓練/data/f03/label"
    ]

    dataset = JsonSpineDataset(img_json_files, label_json_files, img_roots, label_roots,
                               target_w=TARGET_W, target_h=TARGET_H, use_aug=USE_AUG)
    print("Dataset usable size:", len(dataset))

    requested_splits = 3
    n_samples = len(dataset)
    if n_samples < requested_splits:
        n_splits = max(2, n_samples)
        print(f"[Warning] dataset has only {n_samples} samples; set n_splits={n_splits}")
    else:
        n_splits = requested_splits

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # pos_weight for bone BCE（用全資料集粗估；實作中又會用訓練集重算一次）
    pw_value = compute_pos_weight(dataset)
    pw_value = max(pw_value, 1.0)
    pos_weight_tensor_global = torch.tensor([pw_value], dtype=torch.float32).to(device)

    fold_results, best_model_paths, all_histories = [], [], []

    for fold, (train_idx, val_idx) in enumerate(kfold.split(dataset)):
        print(f"\n===== Fold {fold+1}/{n_splits} =====")
        train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True,
                                  num_workers=NUM_WORKERS, pin_memory=False)
        val_loader = DataLoader(Subset(dataset, val_idx), batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=NUM_WORKERS, pin_memory=False)

        # 用訓練集重算 pos_weight
        pw_value = compute_pos_weight(Subset(dataset, train_idx))
        pw_value = max(pw_value, 1.0)
        pos_weight_tensor = torch.tensor([pw_value], dtype=torch.float32).to(device)
        criterion_bce_logits = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

        model = UNetWithCountBoundary(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                                      in_channels=IN_CHANNELS, out_classes=2).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR)

        model, best_dice, history = train_model(
            model,
            train_loader,
            val_loader,
            optimizer,
            criterion_bce_logits,
            epochs=EPOCHS,
            device=device,
        )
        fold_results.append(best_dice)
        all_histories.append(history)

        # save fold best
        model_path = os.path.join(OUTPUT_DIR, f"unet_best_fold{fold+1}.pth")
        torch.save(model.state_dict(), model_path)
        best_model_paths.append(model_path)
        print(f"Fold {fold+1} best model saved to: {model_path}")

        # plots per fold
        epochs_range = list(range(1, len(history['train_loss']) + 1))
        plt.figure(); plt.plot(epochs_range, history['train_loss']); plt.plot(epochs_range, history['val_loss'])
        plt.title(f"Fold{fold+1} Loss"); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.legend(["train_loss", "val_loss"]); plt.savefig(os.path.join(PLOTS_DIR, f"fold{fold+1}_loss.png")); plt.close()

        plt.figure(); plt.plot(epochs_range, history['train_dice']); plt.plot(epochs_range, history['val_dice'])
        plt.title(f"Fold{fold+1} Dice"); plt.xlabel("Epoch"); plt.ylabel("Dice")
        plt.legend(["train_dice", "val_dice"]); plt.savefig(os.path.join(PLOTS_DIR, f"fold{fold+1}_dice.png")); plt.close()
        print(f"[Plots] saved fold{fold+1} plots to {PLOTS_DIR}")

        # optional finetune
        if FINETUNE:
            try:
                print("Starting finetune stage...")
                model_ft = UNetWithCountBoundary(encoder_name=ENCODER_NAME, encoder_weights=ENCODER_WEIGHTS,
                                                 in_channels=IN_CHANNELS, out_classes=2).to(device)
                ckpt = os.path.join(OUTPUT_DIR, "unet_best_current_run.pth")
                if os.path.exists(ckpt):
                    model_ft, best_ft, history_ft = finetune_from_checkpoint(
                        model_ft, train_loader, val_loader, ckpt, device, LR,
                        factor=FINETUNE_LR_FACTOR,
                        epochs=FINETUNE_EPOCHS, patience=FINETUNE_PATIENCE,
                        pos_weight_bone=pos_weight_tensor
                    )
                    ft_path = os.path.join(OUTPUT_DIR, f"unet_finetuned_fold{fold+1}.pth")
                    torch.save(model_ft.state_dict(), ft_path)
                    print(f"Finetuned model saved to: {ft_path} (best val dice during finetune {best_ft:.4f})")
                else:
                    print("No checkpoint found for finetune, skipping.")
            except Exception as e:
                print("Finetune stage error:", e)

    print("\nK-Fold results:", fold_results)
    if len(fold_results) > 0:
        print("Average Dice:", float(np.mean(fold_results)))
        best_fold_idx = int(np.argmax(fold_results))
        print("Overall best fold: {}, model path: {}".format(best_fold_idx+1, best_model_paths[best_fold_idx]))
        overall_best = os.path.join(OUTPUT_DIR, "unet_best_overall.pth")
        torch.save(torch.load(best_model_paths[best_fold_idx], map_location='cpu'), overall_best)
        print("Overall best model saved to:", overall_best)

    # aggregated CV plots
    if len(all_histories) > 0:
        ensure_dir(PLOTS_DIR)
        plt.figure()
        for h in all_histories:
            plt.plot(range(1, len(h['val_loss'])+1), h['val_loss'], alpha=0.4)
        plt.title("Validation Loss - all folds"); plt.xlabel("Epoch"); plt.ylabel("Val Loss")
        plt.savefig(os.path.join(PLOTS_DIR, "cv_val_loss_allfolds.png")); plt.close()

        plt.figure()
        for h in all_histories:
            plt.plot(range(1, len(h['val_dice'])+1), h['val_dice'], alpha=0.4)
        plt.title("Validation Dice - all folds"); plt.xlabel("Epoch"); plt.ylabel("Val Dice")
        plt.savefig(os.path.join(PLOTS_DIR, "cv_val_dice_allfolds.png")); plt.close()
        print(f"[Plots] saved CV aggregated plots to {PLOTS_DIR}")

    # Ensemble evaluation top-K folds
    print("[Ensemble] Selecting top-K folds for ensemble...")
    sorted_idx = np.argsort(fold_results)[::-1]
    topk_idx = sorted_idx[:ENSEMBLE_TOPK]
    topk_paths = [best_model_paths[i] for i in topk_idx]
    print(f"[Ensemble] top-{ENSEMBLE_TOPK} folds: {topk_paths}")

    best_t, best_d_before, mean_dice_after, per_dices = evaluate_ensemble_on_dataset(
        topk_paths, dataset, device, save_results_dir=ENSEMBLE_DIR
    )
    print(f"[Ensemble final] best threshold={best_t:.3f}, dice before postprocess={best_d_before:.4f}, "
          f"dice after postprocess={mean_dice_after:.4f}")
