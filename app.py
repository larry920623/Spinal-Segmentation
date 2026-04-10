#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
app.py - Vertebra segmentation demo aligned with a.py (UNetWithCountBoundary)
- bone + boundary 兩通道
- boundary 引導 + 距離轉換 + watershed 做實例分割
- 大塊由 boundary profile 水平切段 (max area split)
- ✅ 新增：避免重疊 + 自動補齊漏偵測的一節（依中位間距與中位尺寸）
"""

import io, base64
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, request, redirect
import segmentation_models_pytorch as smp

# ---------------- CONFIG ----------------
MODEL_PATH   = r"C:\Users\user\spinal\Spinal-Segmentation\unet_best_current_run.pth"
ENCODER_NAME = "resnet34"
TARGET_W, TARGET_H = 224, 512
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DEVICE = torch.device('cpu')

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
GAP_FACTOR_FOR_MISSING = 1.60     # 縱向中心距過大判定倍率
NEWBOX_MAX_PER_GAP     = 3        # 一個大間隙最多補幾個
BOX_H_MIN_RATIO        = 0.55     # 調整造成太矮時，至少維持 median_h * 0.55
CLAMP_W_RANGE          = (26, 80) # 對中位寬度做合理夾取，避免極端
CLAMP_H_RANGE          = (20, 90) # 對中位高度做合理夾取

X_ALIGN_DELTA = 14      # 允許每個框的中心相對「脊柱中心線」的最大偏差
WIDTH_CLAMP_RATIO = (0.7, 1.4)  # 保留原始寬，但限制在 median_w 的 ±70%~140%

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
        dec = self.unet.decoder(feats)
        logits2 = self.unet.segmentation_head(dec)  # (B,2,H,W)
        c = self.count_head(deepest)
        c = torch.flatten(c, 1)
        count_pred = self.count_fc(c)
        return logits2, count_pred


def load_trained_model(weight_path, device=DEVICE, encoder_name=ENCODER_NAME):
    model = UNetWithCountBoundary(encoder_name=encoder_name, encoder_weights=None,
                                  in_channels=1, out_classes=2).to(device)
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

MODEL = load_trained_model(MODEL_PATH, DEVICE, ENCODER_NAME)

# ---------------- Preprocessing ----------------
def preprocess_image(img_pil):
    img_gray = np.array(img_pil.convert("L"))
    img_gray = cv2.resize(img_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_AREA)
    img_f = (img_gray.astype(np.float32) / 255.0)[None, None, :, :]
    return torch.tensor(img_f, dtype=torch.float32), img_gray

# ---------------- Utils ----------------
def dice_np(pred, mask, eps=1e-6):
    pred_b = (pred > 0).astype(np.uint8)
    mask_b = (mask > 0).astype(np.uint8)
    inter = int((pred_b & mask_b).sum())
    denom = int(pred_b.sum() + mask_b.sum())
    return float((2*inter + eps) / (denom + eps))

def best_dice_over_thresholds(prob_map, gt_mask_255, ths=(0.1, 0.3, 0.5, 0.7, 0.9)):
    """
    prob_map: float32 (H,W) in [0,1], 模型的 bone probability
    gt_mask_255: uint8 (H,W) with {0,255}
    ths: 要測的 thresholds
    return: best_thr, best_dice, results(list of (thr, dice))
    """
    gt_bin = (gt_mask_255 > 0).astype(np.uint8)
    results = []
    for t in ths:
        pred_bin = (prob_map >= t).astype(np.uint8)
        d = dice_np(pred_bin, gt_bin)
        results.append((t, d))
    best_thr, best_dice = max(results, key=lambda x: x[1])
    return best_thr, best_dice, results


def bgr_to_data_uri(img_bgr):
    ok, buf = cv2.imencode(".png", img_bgr)
    b64 = base64.b64encode(buf).decode("ascii")
    return f"data:image/png;base64,{b64}"

def draw_boxes_and_centroids(base_img_gray, boxes, cents,
                             color_box=(0,0,255), color_cent=(0,255,0)):
    img = cv2.cvtColor(base_img_gray, cv2.COLOR_GRAY2BGR)
    for (x1,y1,x2,y2) in boxes:
        cv2.rectangle(img, (x1,y1), (x2,y2), color_box, 2)
    for (cx,cy) in cents:
        cv2.circle(img, (cx,cy), 4, color_cent, -1)
    return img

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
        if (x2-x1)*(y2-y1) > DROP_TOO_BIG_AFTER_SPLIT:  # 丟掉極端
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
    """boxes 已依 y 排序；用相鄰中心中點當分界，讓 y 範圍互不重疊"""
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
        # 若調整後太矮，維持最小高度
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
    """對相鄰中心距過大的地方插入新框；boxes 已依 y 排序"""
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
        # 需要補框
        n_new = int(np.clip(round(gap / step_med) - 1, 1, NEWBOX_MAX_PER_GAP))
        y_start = cents[i][1]
        step = gap / (n_new + 1)
        for k in range(n_new):
            cy = int(round(y_start + (k+1)*step))
            h = int(round(med_h))
            w = int(round(med_w))
            y1 = int(np.clip(cy - h//2, 0, H-1))
            y2 = int(np.clip(y1 + h, 0, H))
            # 以共同中心線建立新框
            x1 = int(np.clip(x_med - w//2, 0, TARGET_W-1))
            x2 = int(np.clip(x1 + w, 0, TARGET_W))
            added.append((x1,y1,x2,y2))
    if added:
        out = sorted(out + added, key=lambda b: (b[1]+b[3])//2)
    return out

def harmonize_and_fix_boxes(boxes, imgW, imgH):
    """
    不做任何 x 方向置中調整，完全保留模型預測的中心與寬度。
    只做：
      1) 補漏：用中位間距與尺寸在大間隙插入新框
      2) 去重疊：相鄰中心中點切分
      3) 最小高度限制：避免過薄
    """
    if not boxes:
        return boxes

    # 計算共同的中位寬高 (僅用來補漏 & 限制最小高度)
    med_w, med_h, x_med = _median_dims_and_centerline(boxes, imgW, imgH)

    # 保留模型原始預測，不調整 x 與寬
    aligned = sorted(boxes, key=lambda b: (b[1] + b[3]) // 2)

    # 1) 補漏（大間隙）
    filled = fill_missing_boxes_sorted(aligned, med_w, med_h, x_med, imgH)

    # 2) 去重疊
    nonoverlap = enforce_non_overlap_sorted(filled, med_h, imgH)

    # 3) 最小高度限制
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


# ===================================================================


# ---------------- Inference ----------------
def infer_and_extract(img_tensor):
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

    # 先處理：大塊再切
    boxes = split_large_components_by_boundary(boxes, bnd_prob)
    # 仍偏高的再均分
    boxes = vertical_split_tall_boxes(boxes)

    # ✅ 新增：幾何後處理 → 補漏 + 去重疊
    boxes = harmonize_and_fix_boxes(boxes, TARGET_W, TARGET_H)

    cents = [((b[0]+b[2])//2, (b[1]+b[3])//2) for b in boxes]
    return bone_prob, bnd_prob, boxes, cents, count_val


# ---------------- Flask app ----------------
app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def upload_file():
    if request.method == "POST":
        if "image" not in request.files or "gt" not in request.files:
            return redirect(request.url)
        file_img = request.files["image"]
        file_gt = request.files["gt"]
        if file_img.filename == "" or file_gt.filename == "":
            return redirect(request.url)

        img_pil = Image.open(io.BytesIO(file_img.read()))
        gt_pil  = Image.open(io.BytesIO(file_gt.read()))

        img_tensor, img_gray = preprocess_image(img_pil)

        gt_gray = np.array(gt_pil.convert("L"))
        gt_resized = cv2.resize(gt_gray, (TARGET_W, TARGET_H), interpolation=cv2.INTER_NEAREST)
        gt_mask = (gt_resized > 127).astype(np.uint8) * 255

        bone_prob, bnd_prob, boxes, cents, count_head = infer_and_extract(img_tensor)

        pred_mask = (bone_prob >= BONE_THR).astype(np.uint8) * 255
        dice_val = dice_np(pred_mask, gt_mask)
        # --- 新增：掃 5 個 threshold 的 dice，並取最佳 ---
        best_thr, best_dice, dice_list = best_dice_over_thresholds(
            bone_prob, gt_mask, ths=(0.1, 0.3, 0.5, 0.7, 0.9)
        )
        dice_table_html = "<ul>" + "".join([f"<li>thr={t:.1f} → Dice={d:.4f}</li>" for t,d in dice_list]) + "</ul>"


        num_labels, stats = cv2.connectedComponentsWithStats((gt_mask>0).astype(np.uint8), connectivity=8)[:2]
        gt_count = max(0, num_labels-1)

        pred_overlay = draw_boxes_and_centroids(img_gray, boxes, cents, (0,0,255), (0,255,0))

        gt_boxes, gt_cents = [], []
        nlab, labels_cc, stats_cc, cents_gt = cv2.connectedComponentsWithStats((gt_mask>0).astype(np.uint8), 8)
        for i in range(1, nlab):
            x,y,w,h,area = stats_cc[i]
            gt_boxes.append((x,y,x+w,y+h))
            gt_cents.append((int(cents_gt[i][0]), int(cents_gt[i][1])))
        gt_overlay = draw_boxes_and_centroids(img_gray, gt_boxes, gt_cents, (255,0,0), (0,255,255))

        img_pred_uri = bgr_to_data_uri(pred_overlay)
        img_gt_uri   = bgr_to_data_uri(gt_overlay)

        return f"""
        <!doctype html>
        <title>Result</title>
        <h2>Segmentation Result (Boundary-guided + MaxArea Split + Anti-Overlap + Fill-Missing)</h2>
        <p>Dice = {dice_val:.4f}</p>
        <p><b>Best Dice (over 0.1/0.3/0.5/0.7/0.9)</b> = {best_dice:.4f} at thr = {best_thr:.1f}</p>
        <p>All thresholds:</p>
        {dice_table_html}
        <p>Pred vertebrae (instances) = {len(boxes)} | GT vertebrae = {gt_count} | Count head (raw) = {count_head:.2f}</p>
        <h3>Prediction (red boxes, green dots)</h3>
        <img src="{img_pred_uri}" width="380"><br>
        <h3>GT (blue boxes, yellow dots)</h3>
        <img src="{img_gt_uri}" width="380"><br><br>
        <a href="/">&#8592; Back</a>
        """

    return """
    <!doctype html>
    <title>Upload Image + GT</title>
    <h1>Upload Image and GT</h1>
    <form method=post enctype=multipart/form-data>
      Image: <input type=file name=image><br><br>
      GT mask: <input type=file name=gt><br><br>
      <input type=submit value=Upload>
    </form>
    """

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
