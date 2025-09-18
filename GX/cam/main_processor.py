# main_processor.py
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

# ===================================================================
# 1. 统一配置
# ===================================================================
class CONFIG:
    # --- 图像捕捉配置 ---
    CAPTURE_SAVE_DIR = Path('captured_images')
    BLUR_THRESHOLD = 800
    EXPECTED_SIZE = (800, 800)
    REQUIRED_ID_ORDER = [0, 1, 2, 3]
    CORNER_MODE = 'outer'

    # --- 图像分析配置 ---
    ANALYSIS_OUTPUT_DIR = Path("outputs")
    TOP_FRACTION_Y = 4/7
    YOLO_MODEL_PATH = "yolov8x.pt"

# 确保文件夹存在
CONFIG.CAPTURE_SAVE_DIR.mkdir(exist_ok=True)
CONFIG.ANALYSIS_OUTPUT_DIR.mkdir(exist_ok=True)


# ===================================================================
# 2. 核心函数
# ===================================================================

def load_yolo_model():
    """加载并返回YOLO模型对象"""
    try:
        model = YOLO(CONFIG.YOLO_MODEL_PATH)
        model.predict(np.zeros((640, 640, 3)), verbose=False) 
        return model
    except Exception as e:
        print(f"错误: YOLO模型加载失败: {e}")
        return None

def _process_and_warp_frame(raw_frame):
    # 拉普拉斯清晰度检查
    gray_blur = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    clarity_score = cv2.Laplacian(gray_blur, cv2.CV_64F).var()
    if clarity_score < CONFIG.BLUR_THRESHOLD:
        return None, f"图像模糊 (Score: {clarity_score:.0f})"

    # ArUco 检测
    gray_aruco = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    try:
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        aruco_params = cv2.aruco.DetectorParameters()
        corners, ids, _ = cv2.aruco.detectMarkers(gray_aruco, aruco_dict, parameters=aruco_params)
    except Exception as e:
        return None, f"ArUco检测失败: {e}"

    if ids is None or any(req not in ids.flatten() for req in CONFIG.REQUIRED_ID_ORDER):
        return None, "未检测到所有必需的ArUco标记"

    # 提取角点
    ids_flat = ids.flatten().tolist()
    id_to_idx = {int(v): i for i, v in enumerate(ids_flat)}
    all_centers = [np.asarray(c).reshape(-1, 2).mean(axis=0) for c in corners]
    overall_centroid = np.mean(np.vstack(all_centers), axis=0)
    
    src_pts_list = []
    for req_id in CONFIG.REQUIRED_ID_ORDER:
        if req_id not in id_to_idx: return None, "逻辑错误: 缺少ID"
        idx = id_to_idx[req_id]
        pts = np.asarray(corners[idx]).reshape(-1, 2).astype(np.float32)
        d = np.linalg.norm(pts - overall_centroid, axis=1) # 'outer' mode
        src_pts_list.append(pts[int(np.argmax(d))])
        
    src_pts = np.array(src_pts_list, dtype=np.float32)

    # 透视变换
    W, H = int(CONFIG.EXPECTED_SIZE[0]), int(CONFIG.EXPECTED_SIZE[1])
    dst_pts = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)
    h_mat, _ = cv2.findHomography(src_pts, dst_pts, method=cv2.RANSAC, ransacReprojThreshold=5.0)

    if h_mat is None:
        return None, "单应性矩阵计算失败"

    warped = cv2.warpPerspective(raw_frame, h_mat, (W, H))
    return warped, "成功提取目标区域"

def _analyze_warped_image(image_bgr, yolo_model):
    """对已裁切的图片进行QR和YOLO分析"""
    # 1) 读二维码
    detector = cv2.QRCodeDetector()
    qr_text, _, _ = detector.detectAndDecode(image_bgr)
    
    # 2) YOLO检测
    res = yolo_model.predict(image_bgr, conf=0.25, verbose=False)[0]
    im_h = res.orig_img.shape[0]
    y_thresh = im_h * float(CONFIG.TOP_FRACTION_Y)
    
    best_detection = None
    for b in res.boxes:
        xyxy = b.xyxy.cpu().numpy().astype(int)[0]
        conf = float(b.conf.cpu().numpy()[0])
        cls_id = int(b.cls.cpu().numpy()[0])
        _, y1, _, y2 = xyxy
        cy = (y1 + y2) / 2
        if cy < y_thresh:
            area = (xyxy[2] - xyxy[0]) * (xyxy[3] - xyxy[1])
            if best_detection is None or area > best_detection[0]:
                cls_name = res.names.get(cls_id, str(cls_id))
                best_detection = (area, cls_name, conf)
    
    detection_result = None
    if best_detection:
        _, cls_name, conf = best_detection
        detection_result = {"class_name": cls_name, "confidence": conf}

    return {
        "qr_code": qr_text if qr_text else "N/A",
        "detection": detection_result
    }


def run_analysis_on_frame(raw_frame, yolo_model):
    """
    对外暴露的主接口函数：执行完整的“预处理+分析”流程。
    接收一个原始帧，返回 (分析结果字典 | None, 状态信息字符串)。
    """
    warped_image, status = _process_and_warp_frame(raw_frame)
    
    if warped_image is not None:
        analysis_result = _analyze_warped_image(warped_image, yolo_model)
        return analysis_result, "分析完成"
    else:
        # 如果预处理失败，直接返回失败信息
        return None, status

# 别名
run_one_full_cycle = run_analysis_on_frame