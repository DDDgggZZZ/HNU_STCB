import cv2
import os
import time
import numpy as np


# ===================== 配置 =====================
url = 'http://192.168.5.1:81/stream'  # ESP32-CAM 流地址
save_dir = 'captured_images'
os.makedirs(save_dir, exist_ok=True)

CAPTURE_COOLDOWN = 3.0          # 截图冷却时间（秒）
BLUR_THRESHOLD = 2500           # 清晰度阈值（拉普拉斯方差）
EXPECTED_SIZE = (800, 800)      # 透视展开后的目标尺寸 (W, H)

# 需要检测到的 ArUco ID 及其角位置约定：0:TL, 1:TR, 2:BR, 3:BL
REQUIRED_ID_ORDER = [0, 1, 2, 3]

# 角点选择模式：
#  - 'outer': 取每个标记相对四标记质心“最远”的那个角（默认，更贴近外侧角）
#  - 'inner': 取相对质心“最近”的角（若四标贴在被裁区域内侧，可用这个）
#  - 'center': 直接用标记中心（简单但精度略低）
CORNER_MODE = 'outer'

# ===================== 全局 =====================
last_capture_time = 0.0
frame_count = 0

# ===================== 工具函数 =====================
def check_blur(frame):
    """返回 (清晰度得分, 是否清晰)"""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    score = cv2.Laplacian(gray, cv2.CV_64F).var()
    return score, score > BLUR_THRESHOLD


def _get_aruco_dictionary(name="DICT_4X4_50"):
    if not hasattr(cv2, 'aruco'):
        raise RuntimeError("当前 OpenCV 未编译 aruco 模块。")
    aruco = cv2.aruco
    if hasattr(aruco, name):
        return aruco.getPredefinedDictionary(getattr(aruco, name))
    # 回退
    return aruco.getPredefinedDictionary(aruco.DICT_4X4_50)


def _aruco_detect(gray, dictionary, parameters):
    """兼容老/新版本接口"""
    aruco = cv2.aruco
    if hasattr(aruco, 'ArucoDetector'):
        detector = aruco.ArucoDetector(dictionary, parameters)
        corners, ids, rejected = detector.detectMarkers(gray)
    else:
        corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
    return corners, ids, rejected


def _select_point_from_marker(pts4x2, centroid_xy, mode='outer'):
    """
    从单个 ArUco 的四角中选一个代表角。
    - pts4x2: (4,2) 顺序由 OpenCV 给出，方向随标记朝向变化
    - centroid_xy: 四个标记的总体质心
    - mode: 'outer' | 'inner' | 'center'
    """
    pts = np.asarray(pts4x2, dtype=np.float32)
    if mode == 'center':
        return pts.mean(axis=0)

    d = np.linalg.norm(pts - centroid_xy, axis=1)
    if mode == 'outer':
        return pts[int(np.argmax(d))]
    else:  # 'inner'
        return pts[int(np.argmin(d))]


def _collect_src_points(corners_list, ids_array, id_order, mode='outer'):
    """
    根据 id_order 收集四点（按照 TL,TR,BR,BL 顺序返回）。
    - corners_list: list of (1,4,2) 或 (4,2)
    - ids_array: (N,) 或 (N,1)
    - mode: 选角模式
    """
    ids_flat = ids_array.flatten().tolist()
    id_to_idx = {int(v): i for i, v in enumerate(ids_flat)}

    # 计算四标记总体质心（用于 inner/outer 判断）
    all_centers = []
    for i in range(len(ids_flat)):
        pts = np.asarray(corners_list[i]).reshape(-1, 2)
        c = pts.mean(axis=0)
        all_centers.append(c)
    overall_centroid = np.mean(np.vstack(all_centers), axis=0)

    src_pts = []
    for req_id in id_order:
        if req_id not in id_to_idx:
            return None  # 缺少必要 ID
        idx = id_to_idx[req_id]
        pts = np.asarray(corners_list[idx]).reshape(-1, 2).astype(np.float32)
        p = _select_point_from_marker(pts, overall_centroid, mode=mode)
        src_pts.append(p)
    return np.array(src_pts, dtype=np.float32)  # (4,2) 按 TL,TR,BR,BL


# ===================== 主逻辑：检测并裁剪 =====================
def find_markers_and_capture(raw_frame):
    """
    检测 ArUco 0/1/2/3 -> 选择代表角点 -> 求单应矩阵 -> 只保存透视展开后的矩形区域。
    显示帧含叠加绘制，但保存永远是干净的 warped。
    """
    global last_capture_time, frame_count

    display_frame = raw_frame.copy()
    Ht, Wt, _ = raw_frame.shape

    # 冷却期内仅显示清晰度
    if time.time() - last_capture_time < CAPTURE_COOLDOWN:
        clarity, ok = check_blur(raw_frame)
        cv2.putText(display_frame, f"Clarity: {clarity:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if not ok:
            cv2.putText(display_frame, "Too Blurry", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display_frame

    # 清晰度判断
    clarity, ok = check_blur(raw_frame)
    cv2.putText(display_frame, f"Clarity: {clarity:.2f}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    if not ok:
        cv2.putText(display_frame, "Too Blurry", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display_frame

    # 灰度 + ArUco 检测
    gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
    dic = _get_aruco_dictionary("DICT_4X4_50")
    params = cv2.aruco.DetectorParameters()
    corners, ids, rejected = _aruco_detect(gray, dic, params)

    # 可视化检测到的标记（黄）
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(display_frame, corners, ids, borderColor=(0, 255, 255))

    # 若没全：直接返回
    if ids is None or any(req not in ids.flatten().tolist() for req in REQUIRED_ID_ORDER):
        return display_frame

    # 选取四个角点（按 TL,TR,BR,BL 顺序）
    src = _collect_src_points(corners, ids, REQUIRED_ID_ORDER, mode=CORNER_MODE)
    if src is None:
        return display_frame

    # 目标四点
    W, H = int(EXPECTED_SIZE[0]), int(EXPECTED_SIZE[1])
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

    # 求单应矩阵并透视展开
    H_mat, inliers = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if H_mat is None:
        cv2.putText(display_frame, "Homography failed", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return display_frame

    warped = cv2.warpPerspective(raw_frame, H_mat, (W, H))

    # 叠加显示：把选中的四点连成绿框
    poly = src.reshape(-1, 1, 2).astype(np.int32)
    cv2.polylines(display_frame, [poly], isClosed=True, color=(0, 255, 0), thickness=3)
    for (x, y) in src.astype(int):
        cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
    cv2.putText(display_frame, "Aruco quad locked", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 保存结果（只保存透视展开后的矩形）
    frame_count_plus = False
    try:
        filename = os.path.join(save_dir, f'capture_{frame_count + 1:04d}.jpg')
        if cv2.imwrite(filename, warped):
            frame_count += 1
            frame_count_plus = True
            print(f"检测到 ArUco {REQUIRED_ID_ORDER}，已保存裁剪展开图: {filename}")
            last_capture_time = time.time()
        else:
            print(f"错误: 保存图片失败 {filename}")
    except Exception as e:
        print(f"保存失败: {e}")

    return display_frame


# ===================== 运行 =====================
def main():
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("错误: 无法打开视频流。请检查URL地址或网络连接。")
        return

    cv2.namedWindow('Camera Stream', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Camera Stream', 960, 540)

    print("程序已启动，正在检测 ArUco 角标 0/1/2/3 ...")
    print(f"- 同时检测到 {REQUIRED_ID_ORDER} 且清晰时，将只保存它们确定的矩形区域（{EXPECTED_SIZE[0]}x{EXPECTED_SIZE[1]}）。")
    print(f"- 角点选择模式: {CORNER_MODE}（可改为 'inner' 或 'center'）")
    print(f"- 清晰度阈值 (BLUR_THRESHOLD): {BLUR_THRESHOLD}")
    print("- 按 'q' 键退出程序。")

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            print("警告: 无法获取帧，正在重试...")
            time.sleep(0.05)
            continue

        display_frame = find_markers_and_capture(frame)
        cv2.imshow('Camera Stream', display_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("程序已退出。")


if __name__ == "__main__":
    main()
# import cv2
# import os
# import time
# import numpy as np

# # ↑ 新增：抓静态帧用
# import requests
# from collections import deque

# # ===================== 配置 =====================
# CAM_IP = "192.168.5.1"                    # 你的 ESP32-CAM IP
# STREAM_URL = f"http://{CAM_IP}:81/stream" # 预览流
# SNAP_URL   = f"http://{CAM_IP}/capture"   # 静态抓拍（更清晰）

# save_dir = 'captured_images'
# os.makedirs(save_dir, exist_ok=True)

# CAPTURE_COOLDOWN = 3.0          # 截图冷却时间（秒）
# BLUR_THRESHOLD = 2500           # 清晰度阈值（拉普拉斯方差），用于基本把关
# EXPECTED_SIZE = (800, 800)      # 透视展开后的目标尺寸 (W, H)

# # 连帧选择 + 静止判定（实现要点 #3）
# RECENT_WINDOW = 8               # 维护最近 N 帧，保存时取里面“最清晰”的
# FLOW_THRESHOLD = 0.7            # 光流中位位移阈值（像素），越小越“静止”
# MIN_STABLE_FRAMES = 2           # 可选：连续满足静止的最少帧数（抖动过滤）

# # 需要检测到的 ArUco ID 及其角位置约定：0:TL, 1:TR, 2:BR, 3:BL
# REQUIRED_ID_ORDER = [0, 1, 2, 3]

# # 角点选择模式：
# #  - 'outer': 取每个标记相对四标记质心“最远”的那个角（默认，更贴近外侧角）
# #  - 'inner': 取相对质心“最近”的角（若四标贴在被裁区域内侧，可用这个）
# #  - 'center': 直接用标记中心（简单但精度略低）
# CORNER_MODE = 'outer'

# # ===================== 全局（状态） =====================
# last_capture_time = 0.0
# frame_count = 0

# # 实现要点 #3：状态缓存
# recent_frames = deque(maxlen=RECENT_WINDOW)  # 元素：(clarity_mix, lap_var, frame_bgr)
# prev_gray = None
# stable_counter = 0

# # ===================== 工具函数 =====================
# def laplacian_var(frame):
#     """拉普拉斯方差：与原脚本一致，用于阈值把关"""
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     return cv2.Laplacian(gray, cv2.CV_64F).var()

# def clarity_mix(frame):
#     """
#     连帧里“选最清晰”的评分：拉普拉斯方差 + Tenengrad
#     仅用于挑最好的一帧，不改变 BLUR_THRESHOLD 的语义
#     """
#     g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     lap = cv2.Laplacian(g, cv2.CV_64F).var()
#     sx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
#     sy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
#     ten = (sx**2 + sy**2).mean()
#     return 0.6 * lap + 0.4 * ten

# def scene_motion(prev_gray, gray):
#     """LK 光流，返回中位位移 px；prev_gray 为空时返回大数"""
#     if prev_gray is None:
#         return 999.0
#     p0 = cv2.goodFeaturesToTrack(prev_gray, 150, 0.01, 8)
#     if p0 is None:
#         return 999.0
#     p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None)
#     if p1 is None:
#         return 999.0
#     d = np.linalg.norm((p1 - p0)[st[:, 0] == 1], axis=1)
#     return float(np.median(d)) if len(d) > 0 else 999.0

# def _get_aruco_dictionary(name="DICT_4X4_50"):
#     if not hasattr(cv2, 'aruco'):
#         raise RuntimeError("当前 OpenCV 未编译 aruco 模块。")
#     aruco = cv2.aruco
#     if hasattr(aruco, name):
#         return aruco.getPredefinedDictionary(getattr(aruco, name))
#     return aruco.getPredefinedDictionary(aruco.DICT_4X4_50)

# def _aruco_detect(gray, dictionary, parameters):
#     """兼容老/新版本接口"""
#     aruco = cv2.aruco
#     if hasattr(aruco, 'ArucoDetector'):
#         detector = aruco.ArucoDetector(dictionary, parameters)
#         corners, ids, rejected = detector.detectMarkers(gray)
#     else:
#         corners, ids, rejected = aruco.detectMarkers(gray, dictionary, parameters=parameters)
#     return corners, ids, rejected

# def _select_point_from_marker(pts4x2, centroid_xy, mode='outer'):
#     """
#     从单个 ArUco 的四角中选一个代表角。
#     - pts4x2: (4,2) 顺序由 OpenCV 给出，方向随标记朝向变化
#     - centroid_xy: 四个标记的总体质心
#     - mode: 'outer' | 'inner' | 'center'
#     """
#     pts = np.asarray(pts4x2, dtype=np.float32)
#     if mode == 'center':
#         return pts.mean(axis=0)
#     d = np.linalg.norm(pts - centroid_xy, axis=1)
#     if mode == 'outer':
#         return pts[int(np.argmax(d))]
#     else:  # 'inner'
#         return pts[int(np.argmin(d))]

# def _collect_src_points(corners_list, ids_array, id_order, mode='outer'):
#     """
#     根据 id_order 收集四点（按照 TL,TR,BR,BL 顺序返回）。
#     - corners_list: list of (1,4,2) 或 (4,2)
#     - ids_array: (N,) 或 (N,1)
#     - mode: 选角模式
#     """
#     ids_flat = ids_array.flatten().tolist()
#     id_to_idx = {int(v): i for i, v in enumerate(ids_flat)}

#     # 计算四标记总体质心（用于 inner/outer 判断）
#     all_centers = []
#     for i in range(len(ids_flat)):
#         pts = np.asarray(corners_list[i]).reshape(-1, 2)
#         c = pts.mean(axis=0)
#         all_centers.append(c)
#     overall_centroid = np.mean(np.vstack(all_centers), axis=0)

#     src_pts = []
#     for req_id in id_order:
#         if req_id not in id_to_idx:
#             return None  # 缺少必要 ID
#         idx = id_to_idx[req_id]
#         pts = np.asarray(corners_list[idx]).reshape(-1, 2).astype(np.float32)
#         p = _select_point_from_marker(pts, overall_centroid, mode=mode)
#         src_pts.append(p)
#     return np.array(src_pts, dtype=np.float32)  # (4,2) 按 TL,TR,BR,BL

# def grab_snapshot(url=SNAP_URL, timeout=2.0):
#     """
#     要点 #2：触发保存时改用 /capture 抓取静态单帧（一般更清晰、更高分辨率）
#     """
#     try:
#         r = requests.get(url, timeout=timeout)
#         if r.status_code != 200 or len(r.content) < 1000:
#             return None
#         arr = np.frombuffer(r.content, np.uint8)
#         img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
#         return img
#     except Exception:
#         return None

# # ===================== 主逻辑：检测、稳触发、抓静态图并裁剪 =====================
# def process_frame_and_maybe_capture(raw_frame):
#     """
#     - 用 /stream 的帧做预览、清晰/静止/标记判断
#     - 触发保存时：优先 /capture 抓静态大图进行 warp 保存（要点 #2）
#     - 连帧窗口中选最清晰（要点 #3）
#     """
#     global last_capture_time, frame_count, prev_gray, stable_counter

#     display_frame = raw_frame.copy()
#     Ht, Wt, _ = raw_frame.shape

#     # 计算清晰与光流
#     gray = cv2.cvtColor(raw_frame, cv2.COLOR_BGR2GRAY)
#     lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
#     flow = scene_motion(prev_gray, gray)
#     prev_gray = gray

#     # 连帧缓存（仅用于“选最清晰”）
#     recent_frames.append((clarity_mix(raw_frame), lap_var, raw_frame.copy()))

#     # 显示清晰、光流信息
#     cv2.putText(display_frame, f"LapVar: {lap_var:.0f}", (10, 28),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#     cv2.putText(display_frame, f"Flow: {flow:.2f}px", (10, 56),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # 检测 ArUco
#     gray_for_aruco = gray
#     dic = _get_aruco_dictionary("DICT_4X4_50")
#     params = cv2.aruco.DetectorParameters()
#     corners, ids, _ = _aruco_detect(gray_for_aruco, dic, params)

#     markers_ok = (ids is not None) and all(req in ids.flatten().tolist() for req in REQUIRED_ID_ORDER)

#     # 可视化检测到的标记（黄）
#     if ids is not None and len(ids) > 0:
#         cv2.aruco.drawDetectedMarkers(display_frame, corners, ids, borderColor=(0, 255, 255))

#     # 叠加状态提示
#     if markers_ok:
#         cv2.putText(display_frame, "Markers: OK", (10, 84),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
#     else:
#         cv2.putText(display_frame, "Markers: missing", (10, 84),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # 静止判定计数器
#     if flow < FLOW_THRESHOLD:
#         stable_counter += 1
#     else:
#         stable_counter = 0
#     cv2.putText(display_frame, f"Stable: {stable_counter}/{MIN_STABLE_FRAMES}", (10, 112),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0) if stable_counter >= MIN_STABLE_FRAMES else (0, 0, 255), 2)

#     # 冷却中直接返回（显示 HUD）
#     if time.time() - last_capture_time < CAPTURE_COOLDOWN:
#         return display_frame

#     # 触发条件：四标到齐 + 场景静止 + 至少一个窗口内清晰帧通过阈值
#     trigger = False
#     best_frame = None
#     best_lap = 0.0

#     if markers_ok and stable_counter >= MIN_STABLE_FRAMES and len(recent_frames) > 0:
#         # 找最近窗口里“最清晰”的帧
#         best = max(recent_frames, key=lambda x: x[0])  # 按 clarity_mix
#         _, best_lap, best_frame = best
#         if best_lap > BLUR_THRESHOLD:
#             trigger = True

#     if not trigger:
#         # 没触发就返回（带 HUD）
#         if not markers_ok:
#             cv2.putText(display_frame, "Waiting: need markers 0/1/2/3", (10, 140),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         elif stable_counter < MIN_STABLE_FRAMES:
#             cv2.putText(display_frame, "Waiting: scene not stable", (10, 140),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         else:
#             cv2.putText(display_frame, "Waiting: no clear frame in window", (10, 140),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         return display_frame

#     # 计算当前帧的四点（按 TL,TR,BR,BL）
#     src = _collect_src_points(corners, ids, REQUIRED_ID_ORDER, mode=CORNER_MODE)
#     if src is None:
#         return display_frame

#     # 目标四点
#     W, H = int(EXPECTED_SIZE[0]), int(EXPECTED_SIZE[1])
#     dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], dtype=np.float32)

#     # 求单应矩阵
#     H_mat, _inliers = cv2.findHomography(src, dst, method=cv2.RANSAC, ransacReprojThreshold=2.0)
#     if H_mat is None:
#         cv2.putText(display_frame, "Homography failed", (10, 168),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         return display_frame

#     # 要点 #2：优先用 /capture 静态图做 warp；失败则退回到窗口里最清晰那帧
#     snap = grab_snapshot(SNAP_URL)
#     source_for_warp = snap if snap is not None else best_frame

#     if source_for_warp is None:
#         cv2.putText(display_frame, "No source frame to save", (10, 168),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
#         return display_frame

#     warped = cv2.warpPerspective(source_for_warp, H_mat, (W, H))

#     # 叠加显示：把选中的四点连成绿框
#     poly = src.reshape(-1, 1, 2).astype(np.int32)
#     cv2.polylines(display_frame, [poly], isClosed=True, color=(0, 255, 0), thickness=3)
#     for (x, y) in src.astype(int):
#         cv2.circle(display_frame, (int(x), int(y)), 4, (0, 255, 0), -1)
#     cv2.putText(display_frame, "Aruco quad locked", (10, 196),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

#     # 保存结果（只保存透视展开后的矩形）
#     try:
#         filename = os.path.join(save_dir, f'capture_{frame_count + 1:04d}.jpg')
#         if cv2.imwrite(filename, warped):
#             frame_count += 1
#             print(f"检测到 ArUco {REQUIRED_ID_ORDER}，已保存裁剪展开图: {filename} "
#                   f"(source={'/capture' if snap is not None else 'best_window'})")
#             last_capture_time = time.time()
#             # 触发后清空“稳定计数”和窗口，避免连触发
#             stable_counter = 0
#             recent_frames.clear()
#         else:
#             print(f"错误: 保存图片失败 {filename}")
#     except Exception as e:
#         print(f"保存失败: {e}")

#     return display_frame

# # ===================== 运行 =====================
# def main():
#     cap = cv2.VideoCapture(STREAM_URL)
#     if not cap.isOpened():
#         print("错误: 无法打开视频流。请检查URL地址或网络连接。")
#         return

#     cv2.namedWindow('Camera Stream', cv2.WINDOW_NORMAL)
#     cv2.resizeWindow('Camera Stream', 960, 540)

#     print("程序已启动，正在检测 ArUco 角标 0/1/2/3 ...")
#     print(f"- 清晰阈值 (BLUR_THRESHOLD): {BLUR_THRESHOLD}")
#     print(f"- 触发采用：连帧最清晰 + 场景静止({FLOW_THRESHOLD}px) + /capture 静态抓拍")
#     print("- 按 'q' 键退出程序。")

#     while True:
#         ret, frame = cap.read()
#         if not ret or frame is None:
#             print("警告: 无法获取帧，正在重试...")
#             time.sleep(0.05)
#             continue

#         display_frame = process_frame_and_maybe_capture(frame)
#         cv2.imshow('Camera Stream', display_frame)

#         if cv2.waitKey(20) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print("程序已退出。")

# if __name__ == "__main__":
#     main()
