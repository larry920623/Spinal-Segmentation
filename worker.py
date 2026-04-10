import os
import base64
import io
from PIL import Image
from celery import Celery
from core import preprocess_image, infer_and_extract, calculate_metrics

redis_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/0")
celery_app = Celery("tasks", broker=redis_url, backend=redis_url)

@celery_app.task(name="tasks.segment_spine")
def segment_spine_task(img_b64: str, gt_b64: str):
    # 1. 解碼兩張圖片
    img_bytes = base64.b64decode(img_b64)
    gt_bytes = base64.b64decode(gt_b64)
    
    img_pil = Image.open(io.BytesIO(img_bytes)).convert("L")
    gt_pil = Image.open(io.BytesIO(gt_bytes)).convert("L")
    
    # 2. 執行推論
    img_tensor, _ = preprocess_image(img_pil)
    bone_prob, bnd_prob, clean_boxes, clean_cents, count_head = infer_and_extract(img_tensor)
    
    # 3. 計算 Dice 分數與 GT 數據
    dice_val, gt_count = calculate_metrics(bone_prob, gt_pil)
    
    return {
        "status": "completed",
        "dice_score": round(dice_val, 4),
        "gt_count": gt_count,
        "predicted_instances": len(clean_boxes),
        "count_head_raw": round(count_head, 2),
        "bounding_boxes": clean_boxes,
        "centroids": clean_cents
    }