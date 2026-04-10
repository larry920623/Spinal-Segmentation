from fastapi import FastAPI, UploadFile, File
import base64
from celery.result import AsyncResult
from worker import segment_spine_task, celery_app

app = FastAPI(title="Spine Segmentation API")

@app.post("/api/segment")
async def create_segmentation_task(
    image: UploadFile = File(...), 
    gt: UploadFile = File(...)
):
    # 讀取影像與 GT
    img_bytes = await image.read()
    gt_bytes = await gt.read()
    
    # 轉為 Base64
    img_b64 = base64.b64encode(img_bytes).decode("utf-8")
    gt_b64 = base64.b64encode(gt_bytes).decode("utf-8")
    
    # 將兩個參數發送給 Celery
    task = segment_spine_task.delay(img_b64, gt_b64)
    
    return {"task_id": task.id, "status": "processing"}

@app.get("/api/tasks/{task_id}")
async def get_task_status(task_id: str):
    task_result = AsyncResult(task_id, app=celery_app)
    if task_result.ready():
        return task_result.result
    else:
        return {"task_id": task_id, "status": "pending"}