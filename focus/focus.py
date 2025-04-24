#!/usr/bin/env python3
#
# Before running this sample
# Please ensure 
# 1. you have the necessary package installed.
# ins_py36_dep.sh
# 2. icam-500 is at playing status
# 
# when icam-500 is at playing at >= 5fps, there is another rtsp stream available at port 8550
# rtsp://<ip address>:8550/video
#
# version: 
#  20230116: Added save png and save raw image methods.
#  20250416: Added focus position control using API, capture images at different focus positions
#  20250416: Added Laplacian focus score calculation for focus quality evaluation
#  20250416: Added time_it decorator for function execution time tracking
#  20250416: Optimized image saving performance
#  20250416: Further improved focus performance by removing unnecessary wait time
#  20250416: Added camera parameter optimization to improve read speed


import cv2
import time
import requests
import os
import numpy as np
from tqdm import trange
import csv
import matplotlib.pyplot as plt
import functools
from datetime import datetime
import threading
from queue import Queue
import concurrent.futures


def time_it(func):
    """
    函數裝飾器：記錄函數執行時間。
    Args:
        func: 被裝飾的函數
    Returns:
        包裝後的函數
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Starting {func.__name__}")
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = (end_time - start_time) * 1000
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        print(f"[{timestamp}] Completed {func.__name__}, execution time: {execution_time:.2f} milliseconds")
        return result
    return wrapper

class FocusController:
    def __init__(self):
        """
        FocusController 物件初始化。
        建立圖片儲存佇列與背景儲存線程控制旗標。
        """
        self.image_save_queue = Queue()
        self.save_thread_running = False
        self.save_thread = None

    def image_save_worker(self):
        """
        背景工作線程，負責從佇列中取出圖片並儲存到硬碟。
        """
        while self.save_thread_running:
            try:
                item = self.image_save_queue.get(timeout=1)
                if item is None:
                    continue
                img, output_path, quality = item
                cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, quality])
                self.image_save_queue.task_done()
            except Exception:
                pass

    @time_it
    def set_focus_position(self, position):
        """
        設定相機對焦位置，透過 API 呼叫。
        Args:
            position (int): 對焦位置 (0-1600)
        Returns:
            bool: 成功回傳 True，失敗回傳 False
        """
        try:
            url = "http://127.0.0.1:5000/camera/focus_abs_position"
            files = {"abs_position": (None, str(position))}
            response = requests.post(url, files=files)
            if response.status_code == 200:
                response_data = response.json()
                if response_data.get("code") == 200 and response_data.get("status") == "OK":
                    return True
                else:
                    print(f"設定對焦位置失敗，API 返回: {response_data}")
                    return False
            else:
                print(f"設定對焦位置失敗，錯誤碼: {response.status_code}")
                return False
        except Exception as e:
            print(f"設定對焦位置時發生錯誤: {str(e)}")
            return False

    @time_it
    def optimize_camera(self, cam):
        """
        優化相機參數以提升讀取速度。
        Args:
            cam (cv2.VideoCapture): OpenCV 相機物件
        Returns:
            cv2.VideoCapture: 優化後的相機物件
        """
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_PROP_FPS, 15)
        cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        for _ in range(5):
            cam.read()
        return cam

    @time_it
    def capture_image(self, cam, output_path, save_quality=85, use_background_save=True):
        """
        從相機擷取單一幀並儲存。
        Args:
            cam (cv2.VideoCapture): 相機物件
            output_path (str): 圖片儲存路徑
            save_quality (int): JPEG 壓縮品質 (1-100)
            use_background_save (bool): 是否使用背景儲存
        Returns:
            (img, bool): 圖片與是否成功
        """
        read_start_time = time.time()
        ret, img = cam.read()
        read_end_time = time.time()
        read_time = (read_end_time - read_start_time) * 1000
        print(f"cam.read() execution time: {read_time:.2f} milliseconds")
        if ret:
            if use_background_save:
                self.image_save_queue.put((img, output_path, save_quality))
            else:
                save_start_time = time.time()
                cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, save_quality])
                save_end_time = time.time()
                save_time = (save_end_time - save_start_time) * 1000
                print(f"cv2.imwrite() execution time: {save_time:.2f} milliseconds")
            return img, True
        else:
            print('Failed to capture image')
            return None, False

    @time_it
    def calculate_focus_score(self, image):
        """
        同時計算圖片的 Sobel 與 Laplacian 清晰度分數，並以 Sobel 分數作為主要依據。
        Args:
            image (np.ndarray): 輸入圖片
        Returns:
            float: Sobel 對焦分數（主依據）
            float: Laplacian 對焦分數（僅供參考）
        Sobel 分數：以 Sobel 邊緣強度的方差作為清晰度指標。
        Laplacian 分數：以 Laplacian 邊緣強度的方差作為清晰度指標。
        """
        small = cv2.resize(image, (960, 540))
        if len(small.shape) == 3:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        else:
            gray = small
        # Sobel 清晰度分數
        sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        sobel = np.hypot(sobelx, sobely)
        sobel_score = np.var(sobel)
        # Laplacian 清晰度分數
        laplacian = cv2.Laplacian(gray, cv2.CV_32F)
        laplacian_score = np.var(laplacian)
        # 僅以 sobel_score 作為主依據
        return sobel_score, laplacian_score

    @time_it
    def process_focus_step(self, cam, position, output_dir, save_quality, use_background_save):
        if self.set_focus_position(position):
            time.sleep(1)  # 保持原本等待時間
            output_path = f"{output_dir}/focus_pos_{position:04d}.jpg"
            img, success = self.capture_image(cam, output_path, save_quality, use_background_save)
            if success:
                sobel_score, laplacian_score = self.calculate_focus_score(img)
                print(f"Position: {position}, Sobel Score: {sobel_score:.2f}, Laplacian Score: {laplacian_score:.2f}")
                return position, sobel_score, laplacian_score
        return position, None, None

    @time_it
    def focus_sweep(self, start_pos=0, end_pos=1600, step=100, step_time=1, save_quality=85, use_background_save=True, optimize_cam_params=False):
        """
        進行對焦位置掃描，從 start_pos 到 end_pos，每 step 單位拍攝一張照片，計算對焦分數並儲存。
        Args:
            start_pos (int): 起始對焦位置
            end_pos (int): 結束對焦位置
            step (int): 對焦位置步進值
            step_time (float): 每步等待秒數
            save_quality (int): 圖片儲存品質
            use_background_save (bool): 是否使用背景儲存
            optimize_cam_params (bool): 是否優化相機參數
        Returns:
            (int, float): 最佳對焦位置與分數
        """
        output_dir = os.path.join(os.path.dirname(__file__), "focus_sweep_images")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        focus_positions = []
        focus_scores = []  # Sobel 分數
        laplacian_scores = []  # Laplacian 分數
        if use_background_save:
            self.save_thread_running = True
            self.save_thread = threading.Thread(target=self.image_save_worker)
            self.save_thread.daemon = True
            self.save_thread.start()
            print("Background saving thread started")
        print("Trying to open camera...")
        cam = cv2.VideoCapture(10)
        if not cam.isOpened():
            print("Failed to open camera using index 10, trying with v4l2 path...")
            cam = cv2.VideoCapture("/dev/video10")
        if not cam.isOpened():
            print("Cannot open camera. Check that icam-500 is at playing status.")
            return
        print("Camera opened successfully")
        cam.set(cv2.CAP_PROP_FORMAT, -1)
        if optimize_cam_params:
            print("Optimizing camera parameters...")
            cam = self.optimize_camera(cam)
        print("Camera ready")
        print("Warming up camera...")
        self.set_focus_position(start_pos)
        time.sleep(1)
        for _ in range(5):
            cam.read()
        print("Starting focus sweep")
        for position in trange(start_pos, end_pos + 1, step):
            pos, sobel_score, laplacian_score = self.process_focus_step(cam, position, output_dir, save_quality, use_background_save)
            if sobel_score is not None:
                focus_positions.append(pos)
                focus_scores.append(sobel_score)
                laplacian_scores.append(laplacian_score)
        cam.release()
        print('Focus sweep completed, captured', len(range(start_pos, end_pos + 1, step)), 'images')
        if use_background_save:
            print("Waiting for all images to be saved...")
            self.image_save_queue.join()
            self.save_thread_running = False
            self.save_thread.join(timeout=2)
            print("All images saved.")
        result_csv = f"focus_scores_{save_quality}.csv"
        with open(result_csv, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Focus Position', 'Sobel Score', 'Laplacian Score'])
            for position, sobel, lap in zip(focus_positions, focus_scores, laplacian_scores):
                writer.writerow([position, sobel, lap])
        print(f"Focus scores saved to {result_csv}")
        plt.figure(figsize=(10, 6))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        color1 = 'tab:blue'
        ax1.set_xlabel('Focus Position')
        ax1.set_ylabel('Sobel Score', color=color1)
        ax1.plot(focus_positions, focus_scores, color=color1, label='Sobel Score')
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.grid(True)
        best_idx = np.argmax(focus_scores)
        best_position = focus_positions[best_idx]
        best_score = focus_scores[best_idx]
        ax1.axvline(x=best_position, color='r', linestyle='--')
        ax1.text(best_position + 20, best_score, f'Best Focus: {best_position}\nSobel: {best_score:.2f}', bbox=dict(facecolor='white', alpha=0.8), color=color1)
        # 右側 y 軸顯示 Laplacian
        ax2 = ax1.twinx()
        color2 = 'tab:orange'
        ax2.set_ylabel('Laplacian Score', color=color2)
        ax2.plot(focus_positions, laplacian_scores, color=color2, label='Laplacian Score')
        ax2.tick_params(axis='y', labelcolor=color2)
        # 圖例
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
        plt.title('Focus Position vs. Focus Score')
        filename = f"focus_score_plot_{save_quality}.png"
        plt.savefig(filename)
        print(f"Focus score plot saved to {filename}")
        print(f"Best focus position: {best_position}, Focus Score: {best_score:.2f}")
        return best_position, best_score

    @time_it
    def automate_focus_sweep(self):
        """
        自動化對焦掃描，使用較低圖片品質與背景儲存加速。
        """
        DIFF1 = 100
        pos1, _ = self.focus_sweep(
            start_pos=200,
            end_pos=1400,
            save_quality=70,
            step=DIFF1,
            step_time=0.5,
            use_background_save=True,
            optimize_cam_params=False
        )
        print(f"Setting good focus position to {pos1}")
        self.set_focus_position(pos1)
        DIFF2 = 20
        pos2, _ = self.focus_sweep(
            start_pos=pos1 - DIFF1,
            end_pos=pos1 + DIFF1,
            save_quality=85,
            step=DIFF2,
            step_time=0.5,
            use_background_save=True,
            optimize_cam_params=False
        )
        print(f"Setting best focus position to {pos2}")
        self.set_focus_position(pos2)

if __name__ == '__main__':
    # 建立 FocusController 物件並執行自動化對焦掃描
    controller = FocusController()
    controller.automate_focus_sweep()

