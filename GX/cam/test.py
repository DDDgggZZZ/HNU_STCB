{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "ModuleNotFoundError",
     "evalue": "No module named 'main_processor'",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mModuleNotFoundError\u001b[0m                       Traceback (most recent call last)",
      "Cell \u001b[1;32mIn[2], line 2\u001b[0m\n\u001b[0;32m      1\u001b[0m \u001b[38;5;66;03m# 导入我们封装好的核心处理函数和YOLO模型加载\u001b[39;00m\n\u001b[1;32m----> 2\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01mmain_processor\u001b[39;00m\u001b[38;5;21;01m.\u001b[39;00m\u001b[38;5;21;01mpy\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m run_one_full_cycle\n\u001b[0;32m      3\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21;01multralytics\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;28;01mimport\u001b[39;00m YOLO\n\u001b[0;32m      5\u001b[0m \u001b[38;5;28;01mdef\u001b[39;00m\u001b[38;5;250m \u001b[39m\u001b[38;5;21mstart_my_application\u001b[39m():\n",
      "\u001b[1;31mModuleNotFoundError\u001b[0m: No module named 'main_processor'"
     ]
    }
   ],
   "source": [
    "# 导入我们封装好的核心处理函数和YOLO模型加载\n",
    "from main_processor.py import run_one_full_cycle\n",
    "from ultralytics import YOLO\n",
    "\n",
    "def start_my_application():\n",
    "    print(\"我的主应用已启动。\")\n",
    "    print(\"准备调用视觉处理模块...\")\n",
    "\n",
    "    # 1. 像主脚本一样，先加载模型\n",
    "    print(\"正在加载YOLO模型...\")\n",
    "    model = YOLO(\"yolov8x.pt\")\n",
    "    print(\"模型加载完毕。\")\n",
    "\n",
    "    # 2. 调用核心函数，它会处理摄像头的启动、捕捉和分析\n",
    "    # 注意：这个函数会暂停，直到一张图片被成功捕捉和分析，或者失败\n",
    "    vision_results = run_one_full_cycle(model)\n",
    "\n",
    "    # 3. 检查是否成功获取结果\n",
    "    if vision_results:\n",
    "        print(\"\\n[我的主应用] 成功接收到视觉分析结果:\")\n",
    "        \n",
    "        # 4. 从返回的字典中提取你需要的数据\n",
    "        qr_data = vision_results.get('qr_code', 'DEFAULT_QR_VALUE')\n",
    "        detection_data = vision_results.get('detection')\n",
    "\n",
    "        print(f\"  - 二维码信息: {qr_data}\")\n",
    "\n",
    "        if detection_data:\n",
    "            object_name = detection_data.get('class_name', 'unknown_object')\n",
    "            confidence = detection_data.get('confidence', 0.0)\n",
    "            print(f\"  - 检测到物体: {object_name} (置信度: {confidence:.2f})\")\n",
    "            \n",
    "            # 在这里，你可以用这些变量做任何事\n",
    "            # 例如：\n",
    "            # if object_name == 'person' and qr_data.startswith('PKG_'):\n",
    "            #     print(\"  - 业务逻辑: 检测到包裹被人拿起，准备记录日志...\")\n",
    "            #     # record_log(qr_data, object_name)\n",
    "            # else:\n",
    "            #     print(\"  - 业务逻辑: 未触发特定条件。\")\n",
    "        else:\n",
    "            print(\"  - 未检测到任何物体。\")\n",
    "\n",
    "    else:\n",
    "        print(\"\\n[我的主应用] 视觉处理模块未能返回结果。\")\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    start_my_application()"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "d2l",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.9.21"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
