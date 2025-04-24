import cv2
from mjpeg_streamer import MjpegServer, Stream
from ultralytics import YOLO
import time
import numpy as np
from PIL import Image
import functools
from datetime import datetime
import threading
import queue

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

class YoloPoseApp:
    def __init__(self, model_path, clothes_path, camera_index=10, stream_size=(1080, 1920), stream_quality=60, stream_fps=30):
        self.model = YOLO(model_path)
        self.stream = Stream("mjpeg", size=stream_size, quality=stream_quality, fps=stream_fps)
        self.server = MjpegServer("0.0.0.0", 5050)
        self.server.add_stream(self.stream)
        self.server.start()
        self.cap = cv2.VideoCapture(camera_index)
        self.clothes_img = Image.open(clothes_path).convert("RGBA")
        self.prev_time = time.time()
        self.frame_time_array = [self.prev_time]
        self.max_fps_samples = 10
        self.preprocess_queue = queue.Queue(maxsize=2)
        self.result_queue = queue.Queue(maxsize=2)
        self.raw_queue = queue.Queue(maxsize=2)
        self.stop_event = threading.Event()
        self.infer_thread = threading.Thread(target=self._inference_worker, daemon=True)
        self.display_thread = threading.Thread(target=self._display_worker, daemon=True)
        self.read_thread = threading.Thread(target=self._read_worker, daemon=True)
        self.preprocess_thread = threading.Thread(target=self._preprocess_worker, daemon=True)
        self.infer_thread.start()
        time.sleep(3)  # 等待模型初始化
        self.display_thread.start()
        self.read_thread.start()
        self.preprocess_thread.start()

    def _inference_worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.preprocess_queue.get(timeout=0.1)
                results = self.model(frame)
                self.result_queue.put((frame, results))
            except queue.Empty:
                continue
    
    def _display_worker(self):
        while not self.stop_event.is_set():
            try:
                frame, results = self.result_queue.get(timeout=0.1)
                self._postprocess_and_show(frame, results)
            except queue.Empty:
                continue

    @time_it
    def _postprocess_and_show(self, frame, results):
        # 合併 plot_result 與 convert_to_pil
        annotated_frame = results[0].plot()
        annotated_pil = Image.fromarray(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
        for result in results:
            self._print_summary(result)
            annotated_pil = self._apply_clothes(result, annotated_pil)
        annotated_frame = cv2.cvtColor(np.array(annotated_pil), cv2.COLOR_RGBA2BGR)
        instant_fps, avg_fps = self._update_fps()
        self._draw_fps_and_show(annotated_frame, instant_fps, avg_fps)
        self.save_and_stream_frame(annotated_frame)

    @time_it
    def read_frame(self):
        success, frame = self.cap.read()
        return success, frame

    @time_it
    def preprocess_frame(self, frame):
        frame = cv2.resize(frame, (1080, 1920))
        frame = cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    @time_it
    def run(self):
        try:
            while not self.stop_event.is_set():
                time.sleep(0.1)  # 主執行緒保持簡潔，僅負責監控stop_event
        finally:
            self.stop_event.set()
            self.read_thread.join()
            self.preprocess_thread.join()
            self.infer_thread.join()
            self.display_thread.join()
            self.cap.release()
            cv2.destroyAllWindows()

    def _read_worker(self):
        while not self.stop_event.is_set() and self.cap.isOpened():
            success, frame = self.read_frame()
            if not success:
                break
            try:
                self.raw_queue.put(frame, timeout=0.1)
            except queue.Full:
                continue

    def _preprocess_worker(self):
        while not self.stop_event.is_set():
            try:
                frame = self.raw_queue.get(timeout=0.1)
                pre_frame = self.preprocess_frame(frame)
                self.preprocess_queue.put(pre_frame, timeout=0.1)
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in preprocess: {e}")
                continue

    def _print_summary(self, result):
        summary = result.summary()
        for obj in summary:
            print(f"Object: {obj['name']}, confidence: {obj['confidence']:.2f}")

    def _apply_clothes(self, result, annotated_pil):
        keypoints_data = result.keypoints
        if keypoints_data is not None and len(keypoints_data) > 0:
            kp = keypoints_data.data[0].cpu().numpy()
            if self._is_keypoints_valid(kp):
                top_y = self._calc_clothes_top_y(kp)
                left_outer, right_outer = self._calc_clothes_outer(kp)
                clothes_width = int(np.linalg.norm(left_outer - right_outer))
                clothes_height = self._calc_clothes_height(kp, top_y)
                clothes_resized = self._resize_clothes(clothes_width, clothes_height)
                offset_x, offset_y = self._calc_clothes_offset(left_outer, right_outer, clothes_width, top_y)
                print(f"衣服貼上成功, 位置: ({offset_x}, {offset_y}), 大小: ({clothes_width}, {clothes_height})")
                annotated_pil.paste(clothes_resized, (offset_x, offset_y), clothes_resized)
            else:
                print("Keypoints not valid, using original frame.")
        return annotated_pil

    def _is_keypoints_valid(self, kp):
        required_indices = [5, 6, 7, 8, 11, 12]
        return kp.shape[0] > max(required_indices) and all(np.any(kp[i] != 0) for i in required_indices)

    def _calc_clothes_top_y(self, kp):
        left_eye = kp[1]
        right_eye = kp[2]
        left_shoulder = kp[5]
        right_shoulder = kp[6]
        left_eye_exists = np.any(left_eye != 0)
        right_eye_exists = np.any(right_eye != 0)
        if left_eye_exists and right_eye_exists:
            left_top = (left_eye + left_shoulder) / 2
            right_top = (right_eye + right_shoulder) / 2
            return int((left_top[1] + right_top[1]) / 2)
        elif left_eye_exists:
            left_top = (left_eye + left_shoulder) / 2
            return int(left_top[1])
        elif right_eye_exists:
            right_top = (right_eye + right_shoulder) / 2
            return int(right_top[1])
        else:
            return int((left_shoulder[1] + right_shoulder[1]) / 2)

    def _calc_clothes_outer(self, kp):
        left_shoulder = kp[5]
        right_shoulder = kp[6]
        left_elbow = kp[7]
        right_elbow = kp[8]
        left_outer = (left_shoulder + left_elbow * 2) / 3
        right_outer = (right_shoulder + right_elbow * 2) / 3
        return left_outer, right_outer

    def _calc_clothes_height(self, kp, top_y):
        hip_center = (kp[11] + kp[12]) / 2
        return int(hip_center[1] - top_y)

    def _resize_clothes(self, width, height):
        return self.clothes_img.resize((width, height))

    def _calc_clothes_offset(self, left_outer, right_outer, width, top_y):
        offset_x = int((left_outer[0] + right_outer[0]) / 2 - width / 2)
        offset_y = int(top_y)
        return offset_x, offset_y

    def _update_fps(self):
        curr_time = time.time()
        self.frame_time_array.append(curr_time)
        if len(self.frame_time_array) > self.max_fps_samples:
            self.frame_time_array.pop(0)
        instant_fps = 1 / (curr_time - self.prev_time)
        self.prev_time = curr_time
        avg_fps = 1 / np.mean(np.diff(self.frame_time_array))
        print(f"Instant FPS: {instant_fps:.2f}, Avg FPS: {avg_fps:.2f}")
        return instant_fps, avg_fps

    def _draw_fps_and_show(self, frame, instant_fps, avg_fps):
        cv2.putText(frame, f"FPS: {instant_fps:.2f}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        cv2.putText(frame, f"Avg FPS: {avg_fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 2)
        # 可加上 cv2.imshow 或其他顯示功能（如有需要）

    def save_and_stream_frame(self, frame, path="output.jpg"):
        # cv2.imwrite(path, frame)
        self.stream.set_frame(frame)

if __name__ == "__main__":
    # 建立主應用物件，指定模型、衣服圖、攝影機等參數
    app = YoloPoseApp(
        model_path="yolo11m-pose.pt",
        clothes_path="tshirt1.png",
        camera_index=10,
        stream_size=(1080, 1920),
        stream_quality=75,
        stream_fps=30
    )
    app.run()