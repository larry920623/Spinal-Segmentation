# 脊椎影像分割與計數：多任務微服務架構 (Multi-Task Framework for Spinal Segmentation & Counting)

##  專案簡介
[cite_start]這是一個端到端 (End-to-End)、非同步的微服務系統，專為醫療影像中的脊椎自動分割與計數而設計 [cite: 52, 53, 56][cite_start]。本專案將深度學習研究與可落地的產品部署相結合，利用多任務 U-Net 架構，以自動化流程提升臨床診斷效率 [cite: 56, 57, 59]。

##  技術堆疊 (Tech Stack)
* [cite_start]**深度學習：** PyTorch, Segmentation Models PyTorch (SMP), OpenCV, NumPy [cite: 60, 105]
* [cite_start]**後端框架：** FastAPI [cite: 107]
* [cite_start]**任務佇列 (Task Queue)：** Celery [cite: 108]
* [cite_start]**訊息代理 (Message Broker)：** Redis [cite: 109]
* [cite_start]**基礎架構與部署：** Docker, Docker Compose (透過 NVIDIA Container Toolkit 支援 GPU 加速) [cite: 106]

##  模型架構 (Model Architecture)
[cite_start]本系統採用多任務學習 (Multi-Task Learning) 策略，同時進行脊椎的分割與數量統計，確保高空間精準度與全局特徵理解 [cite: 52, 53, 79, 86]。
* [cite_start]**核心架構：** U-Net [cite: 59]
* [cite_start]**骨幹網路 (Backbone)：** ResNet34 (使用 ImageNet 預訓練權重) [cite: 61, 67, 73]
* **多任務分支 (Multi-Task Heads)：**
    * [cite_start]**分割分支 (Segmentation Head, ch0)：** 預測核心的骨骼遮罩 (Bone Mask) [cite: 84, 86]。
    * [cite_start]**邊界分支 (Boundary Head, ch1)：** 預測邊界遮罩，用於精確分離相鄰的脊椎實例 [cite: 87]。
    * [cite_start]**計數分支 (Count Head)：** 一個迴歸網路 (Conv2d + Pooling + Linear Layer)，用於預測脊椎的全局總數 [cite: 79, 80, 81, 82]。

##  系統部署架構 (System Architecture)
[cite_start]為避免 API 在繁重的 AI 推論過程中發生超時 (Timeout) 阻塞，本系統採用具備高擴展性的**生產者-消費者 (Producer-Consumer)** 微服務架構進行部署 [cite: 105, 116]：
1. [cite_start]**FastAPI (生產者)：** 接收使用者上傳的影像，生成唯一的 `task_id`，並將推論任務推送到佇列中 [cite: 119, 120, 123]。
2. [cite_start]**Redis (訊息代理與後端)：** 作為訊息傳遞的橋樑，將任務分發給 Worker，並暫存最終的預測結果 [cite: 124, 125, 128]。
3. [cite_start]**Celery Worker (消費者)：** 非同步地從 Redis 提取任務，執行影像前處理、PyTorch 模型推論 (支援 CPU/GPU)，並將結果存回共享空間 [cite: 125, 126, 127]。

##  模型成效 (Performance)
* [cite_start]**最佳 Dice Score：** 0.9525 (於信心閾值 thr=0.9 時) [cite: 144, 150, 151]
* [cite_start]**實例計數 (Instance Counting)：** 精確提取並計算脊椎數量 (在測試中完美預測 16 節脊椎，與真實標籤 Ground Truth 一致) [cite: 152]。

##  快速啟動 (Getting Started)

### 環境要求 (Prerequisites)
* [cite_start]Docker 與 Docker Compose [cite: 106]
* *選用項目：* NVIDIA 驅動程式與 NVIDIA Container Toolkit (若需使用 CUDA GPU 加速)

### 安裝與執行步驟
1. **複製專案：**
    ```bash
    git clone [https://github.com/你的帳號/Spinal-Segmentation.git](https://github.com/你的帳號/Spinal-Segmentation.git)
    cd Spinal-Segmentation
    ```

2. **準備模型權重：**
    請確保已將訓練好的模型權重檔 (`unet_best_overall.pth`) 放置於 `output3/` 目錄下。

3. **建置並啟動容器：**
    ```bash
    docker compose up -d --build
    ```

4. **監控 GPU Worker 運行狀態：**
    ```bash
    docker compose logs -f worker
    ```

##  API 使用說明 (API Usage)
系統啟動後，可透過瀏覽器進入互動式的 Swagger UI 文件介面：
**`http://localhost:8080/docs`** *(註：通訊埠可能因你的本地端設定而異, 自行用vscoode port功能轉接)*

* [cite_start]**POST `/api/segment`**：上傳輸入影像 (Input) 與真實標籤 (GT) 以建立分割任務，系統將回傳一個 `task_id` [cite: 111, 112, 119, 120]。
* [cite_start]**GET `/api/tasks/{task_id}`**：查詢任務狀態。若任務完成，將回傳處理狀態以及最終的推論結果 [cite: 132, 139, 140]。
