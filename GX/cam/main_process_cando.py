import cv2
import os
import time
import numpy as np
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ===================================================================
# 1. 统一配置
# ===================================================================

# --- 摄像头与捕捉配置 ---
ESP32_URL = 'http://192.168.5.1:81/stream'  # ESP32-CAM 流地址
CAPTURE_SAVE_DIR = Path('captured_images')
CAPTURE_COOLDOWN = 3.0       # 截图冷却时间（秒）
BLUR_THRESHOLD = 1200        # 清晰度阈值（拉普拉斯方差）
EXPECTED_SIZE = (800, 800)   # 透视展开后的目标尺寸 (W, H)
REQUIRED_ID_ORDER = [0, 1, 2, 3] # 需要检测到的 ArUco ID
CORNER_MODE = 'outer'        # 角点选择模式: 'outer', 'inner', 'center'

# --- 图像分析配置 ---
ANALYSIS_OUTPUT_DIR = Path("outputs")
TOP_FRACTION_Y = 4/7 # 上侧 4/7 区域用于物体检测

# ===================================================================
# 2. 图像捕捉功能 (来自 location.py)
# ===================================================================
# 工具函数基本不变
def check_blur(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, score > BLUR_THRESHOLD

def _get_aruco_dictionary(name="DICT_4X4_50"):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError("当前 OpenCV 未编译 aruco 模块。")
    aruco = cv2.aruco
    return aruco.getPredefinedDictionary(getattr(aruco, name))

def _aruco_detect(gray, dictionary, parameters):
    aruco = cv2.aruco
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids, rejected

def _select_point_from_marker(pts4x2, centroid_xy, mode='outer'):
    pts = np.asarray(pts4x2, dtype=np.float32)
    if mode == 'center':
        return pts.mean(axis=0)
    d = np.linalg.norm(pts - centroid_xy, axis=1)
    if mode == 'outer':
        return pts[int(np.argmax(d))]
    else:  # 'inner'
        return pts[int(np.argmin(d))]

def _collect_src_points(corners_list, ids_array, id_order, mode='outer'):
    ids_flat = ids_array.flatten().tolist()
    id_to_idx = {int(v): i for i, v in enumerate(ids_flat)}
    all_centers = [np.asarray(c).reshape(-1, 2).mean(axis=0) for c in corners_list]
    overall_centroid = np.mean(np.vstack(all_centers), axis=0)
    src_pts = []
    for req_id in id_order:
        if req_id not in id_to_idx:
            return None
        idx = id_to_idx[req_id]
        pts = np.asarray(corners_list[idx]).reshape(-1, 2).astype(np.float32)
        p = _select_point_from_marker(pts, overall_centroid, mode=mode)
        src_pts.append(p)
    return np.array(src_pts, dtype=np.float32)

# 主逻辑：修改为处理单帧并返回结果
def process_frame_for_capture(raw_frame, frame_count):
    """处理单帧，如果成功则保存图片并返回文件路径，否则返回None"""
    clarity, is_clear = check_blur(raw_frame)
    if not is_clear:
        return None, "Too Blurry"

    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    dic = _get_aruco_dictionary("DICT_4X4_50")
    params = cv2.aruco.DetectorParameters()
    corners, ids, _ = _aruco_detect(gray, dic, params)

    if ids is None or any(req not in ids.flatten() for req in REQUIRED_ID_ORDER):
        return None, "Aruco markers not all found"

    src = _collect_src_points(corners, ids, REQUIRED_ID_ORDER, mode=CORNER_MODE)
    if src is None:
        return None, "Failed to collect source points"

    W, H = int(EXPECTED_SIZE[0]), int(EXPECTED_SIZE[1])
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    H_mat, _ = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    
    if H_mat is None:
        return None, "Homography failed"

    warped = cv2.warpPerspective(raw_frame, H_mat, (W, H))
    
    try:
        filename = CAPTURE_SAVE_DIR / f'capture_{frame_count + 1:04d}.jpg'
        if cv2.imwrite(str(filename), warped):
            print(f"检测到 ArUco {REQUIRED_ID_ORDER}，已保存裁剪展开图: {filename}")
            return str(filename), "Success"
        else:
            return None, f"Error: Failed to save image {filename}"
    except Exception as e:
        return None, f"Save failed: {e}"

# 新增：带开关的捕捉函数
def capture_one_image_on_command():
    """
    启动摄像头，等待直到成功捕捉一张符合条件的图片，然后关闭摄像头。
    返回图片路径或None。
    """
    cap = cv2.VideoCapture(ESP32_URL)
    if not cap.isOpened():
        print("错误: 无法打开视频流。请检查URL地址或网络连接。")
        return None

    print("摄像头已启动，正在寻找目标...")
    
    # 获取当前已有的图片数量，以继续编号
    existing_files = list(CAPTURE_SAVE_DIR.glob('capture_*.jpg'))
    frame_count = len(existing_files)

    start_time = time.time()
    last_status_print_time = time.time()
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.05)
            continue
        
        # 核心处理逻辑
        filepath, status = process_frame_for_capture(frame, frame_count)
        
        # 如果成功，返回路径并关闭摄像头
        if filepath:
            cap.release()
            cv2.destroyAllWindows()
            print("捕捉成功，摄像头已关闭。")
            return filepath
            
        # 每隔2秒打印一次状态，避免刷屏
        if time.time() - last_status_print_time > 2.0:
            print(f"当前状态: {status}")
            last_status_print_time = time.time()

        # 可视化（可选，用于调试）
        cv2.putText(frame, f"Status: {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow('Camera Stream', frame)
        if cv2.waitKey(20) & 0xFF == ord('q'): # 允许手动退出
            break
    
    cap.release()
    cv2.destroyAllWindows()
    print("捕捉失败或手动退出，摄像头已关闭。")
    return None

# ===================================================================
# 3. 图像分析功能 (来自第二段代码)
# ===================================================================
def read_qr(image_bgr):
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image_bgr)
    return data, points

def detect_top_region_object(image_path, model):
    res = model.predict(image_path, conf=0.15, iou=0.6, imgsz=1280, augment=True, verbose=False)[0]
    im_h, im_w = res.orig_img.shape[:2]
    y_thresh = im_h * float(TOP_FRACTION_Y)

    best = None  # (area, xyxy, cls_id, conf)
    for b in res.boxes:
        xyxy = b.xyxy.cpu().numpy().astype(int)[0]
        conf = float(b.conf.cpu().numpy()[0])
        cls_id = int(b.cls.cpu().numpy()[0])
        _, y1, _, y2 = xyxy
        cy = (y1 + y2) / 2

        if cy < y_thresh:
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if best is None or area > best[0]:
                best = (area, xyxy, cls_id, conf)

    if best is None:
        return None

    _, (x1, y1, x2, y2), cls_id, conf = best
    img = Image.open(image_path).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))
    crop_path = ANALYSIS_OUTPUT_DIR / "top_region_crop.png" # 每次覆盖保存
    crop.save(crop_path)

    names = res.names
    cls_name = names.get(cls_id, str(cls_id))
    
    return {
        "class_name": cls_name,
        "confidence": conf,
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "crop_path": str(crop_path),
    }

def analyze_image(image_path, yolo_model):
    """对单张图片进行完整的分析（QR + YOLO）"""
    if not Path(image_path).exists():
        raise FileNotFoundError(f"图片不存在: {image_path}")
        
    img_bgr = cv2.imread(str(image_path))
    
    # 1) 读二维码
    qr_text, _ = read_qr(img_bgr)
    
    # 2) YOLO检测
    detection_result = detect_top_region_object(image_path, yolo_model)
    
    # 3) 整合结果
    final_results = {
        "source_image": image_path,
        "qr_code": qr_text if qr_text else "Not detected",
        "detection": detection_result #可能是None
    }
    
    return final_results

# ===================================================================
# 4. 主流程控制 ("开关")
# ===================================================================
def run_one_full_cycle(yolo_model):
    """
    执行一次完整的“捕捉+分析”流程。
    这是可以被其他脚本调用的核心函数。
    """
    # 第一步：带指令的捕捉
    captured_image_path = capture_one_image_on_command()
    
    # 第二步：如果捕捉成功，自动分析
    if captured_image_path:
        print(f"\n图片捕捉成功: {captured_image_path}")
        print("开始进行图像分析...")
        analysis_results = analyze_image(captured_image_path, yolo_model)
        return analysis_results
    else:
        print("\n未能成功捕捉图片，分析流程终止。")
        return None

if __name__ == "__main__":
    # 创建必要的文件夹
    CAPTURE_SAVE_DIR.mkdir(parents=True, exist_ok=True)
    ANALYSIS_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 预加载YOLO模型，避免每次都加载
    print("正在加载YOLOv8模型，请稍候...")
    yolo = YOLO("yolov8x.pt")
    print("模型加载完毕。")

    print("\n欢迎使用ESP32-CAM自动捕捉与分析系统")
    print("-----------------------------------------")
    
    while True:
        command = input(">> 按 Enter 键开始一次新的[捕捉与分析]，输入 'q' 退出: ")
        if command.lower() == 'q':
            break
        
        # 执行一个完整流程
        final_results = run_one_full_cycle(yolo)
        
        if final_results:
            print("\n✅ === 分析结果 ===")
            print(f"源图片: {final_results['source_image']}")
            print(f"二维码内容: {final_results['qr_code']}")
            
            detection = final_results['detection']
            if detection:
                print("--- 物体检测 ---")
                print(f"  类别: {detection['class_name']}")
                print(f"  置信度: {detection['confidence']:.3f}")
                print(f"  裁切图已保存至: {detection['crop_path']}")
            else:
                print("--- 物体检测 ---")
                print("  未在上部区域检测到目标物体。")
            print("==================\n")
        else:
            print("\n❌ 本次流程未完成。\n")
            
    print("程序已退出。")