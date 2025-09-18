import cv2
from pathlib import Path
from PIL import Image
import numpy as np

# ============ 配置 ============
IMAGE_PATH = "C:\\Users\\29082\\Desktop\\GX\\captured_images\\capture_0005.jpg"  # 你的物流单图片
SAVE_DIR = Path("outputs")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
TOP_FRACTION_Y = 4/7 # 上侧 4/7 区域阈值（仅按纵向 y 判断）

# ============ 1) 二维码检测与读取 ============
def read_qr(image_bgr):
    detector = cv2.QRCodeDetector()
    data, points, _ = detector.detectAndDecode(image_bgr)
    return data, points  # data: str, points: 4x2

# ============ 2)  YOLO 检测 ============
def detect_top_region_object(image_path, top_fraction_y: float = TOP_FRACTION_Y):
    """
    思路：
    - 用 YOLO 在整幅图上做检测（COCO 类别）
    - 过滤出中心点落在“上侧 top_fraction_y 区域”的框（cy < H*top_fraction_y）
    - 选择面积最大的那个框，裁出并保存；返回类别与置信度
    """
    from ultralytics import YOLO  # ultralytics>=8
    model = YOLO("yolov8x.pt")  # 更大的 backbone 精度更好；无则用 yolov8n.pt
    res = model.predict(image_path, conf=0.15, iou=0.6, imgsz=1280, augment=True, verbose=False)[0]
    # model = YOLO("yolov8n.pt")  # COCO 预训练
    # results = model.predict(image_path, conf=0.25, iou=0.5, imgsz=960, verbose=False)
    # res = results[0]

    im_h, im_w = res.orig_img.shape[0], res.orig_img.shape[1]
    y_thresh = im_h * float(top_fraction_y)

    best = None  # (area, xyxy, cls_id, conf)
    for b in res.boxes:
        xyxy = b.xyxy.cpu().numpy().astype(int)[0]      # [x1,y1,x2,y2]
        conf = float(b.conf.cpu().numpy()[0])
        cls_id = int(b.cls.cpu().numpy()[0])
        x1, y1, x2, y2 = xyxy
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2

        # 仅按“上侧 4/7 区域”筛选
        if cy < y_thresh:
            area = max(0, x2 - x1) * max(0, y2 - y1)
            if best is None or area > best[0]:
                best = (area, xyxy, cls_id, conf)

    if best is None:
        return None

    _, (x1, y1, x2, y2), cls_id, conf = best
    # 裁图
    img = Image.open(image_path).convert("RGB")
    crop = img.crop((x1, y1, x2, y2))
    crop_path = SAVE_DIR / "top_region_4of7_crop.png"
    crop.save(crop_path)

    # 输出类别名
    names = res.names
    cls_name = names.get(cls_id, str(cls_id)) if isinstance(names, dict) else (names[cls_id] if cls_id < len(names) else str(cls_id))
    return {
        "class_name": cls_name,
        "confidence": conf,
        "bbox_xyxy": [int(x1), int(y1), int(x2), int(y2)],
        "crop_path": str(crop_path),
    }


def main():
    img_path = Path(IMAGE_PATH)
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path.resolve()}")

    # 1) 读二维码
    img_bgr = cv2.imread(str(img_path))
    qr_text, qr_pts = read_qr(img_bgr)
    print("=== QR 结果 ===")
    print("内容:", qr_text if qr_text else "(未识别到)")
    print("定位点:", qr_pts.tolist() if qr_pts is not None else None)

    # 2) YOLO 检测右上角物体并裁图
    print("\n=== YOLO 检测（上侧 4/7）===")
    det = detect_top_region_object(str(img_path), top_fraction_y=TOP_FRACTION_Y)

    if det is None:
        print("未在右上区域检测到目标。可适当降低 conf 或放宽区域阈值。")
    else:
        print(f"类别: {det['class_name']}, 置信度: {det['confidence']:.3f}")
        print(f"bbox(xyxy): {det['bbox_xyxy']}")
        print(f"裁图已保存: {det['crop_path']}")

if __name__ == "__main__":
    main()
