# Fast Mirror

## 專案簡介
Fast Mirror 是一個基於 FastAPI 的影像處理與分析後端服務，支援自動對焦、YOLO 目標偵測與姿態估計等功能，並可與前端（如 vue-mirror）整合，實現即時影像應用。

## 目錄結構
- `main.py`：專案主入口，啟動 FastAPI 伺服器。
- `tools.py`：輔助工具函式。
- `focus/`：自動對焦相關程式與數據。
- `router/`：API 路由模組。
- `yolo/`：YOLO 模型與推論相關程式與模型檔案。
- `pic/`：測試用圖片與影片。
- `Pipfile`、`Pipfile.lock`：Python 依賴管理。

## 安裝與環境需求
1. 進入 fast_mirror 資料夾
   ```bash
   cd fast_mirror
   ```
2. 安裝 pipenv（如尚未安裝）
   ```bash
   pip install pipenv
   ```
3. 安裝專案依賴
   ```bash
   pipenv install
   ```
4. 啟動虛擬環境
   ```bash
   pipenv shell
   ```

## 啟動方式
1. 啟動 FastAPI 伺服器
   ```bash
   uvicorn main:app --reload
   ```
   或使用 fastapi dev（如有安裝 fastapi CLI）
   ```bash
   fastapi dev main.py
   ```

## 使用說明
- 伺服器啟動後，預設監聽於 http://127.0.0.1:8000
- 可透過 `/docs` 進入 Swagger UI 測試 API

## 主要功能
- 自動對焦分數計算與圖表
- YOLO 目標偵測與姿態估計
- 圖片與影片處理

