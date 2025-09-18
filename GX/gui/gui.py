import sys
import time
import cv2
import numpy as np
import serial
import os
import csv
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
cam_dir = os.path.join(os.path.dirname(current_dir), 'cam')
sys.path.append(cam_dir)
# 从后端模块导入核心功能
from main_processor import run_analysis_on_frame, load_yolo_model
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QTableWidget, QTableWidgetItem, 
                             QLabel, QTextBrowser, QHeaderView, QButtonGroup, QRadioButton,
                             QDialog, QFormLayout, QLineEdit, QMessageBox, QFileDialog)
from PyQt6.QtCore import QThread, pyqtSignal, Qt, QRunnable, QThreadPool, QObject
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor, QPalette, QLinearGradient, QBrush

#识别结果串口定义0xBB +city(0xMN) +month(0xMN)+day(0xMN) + 0x00
dic_city={'ChangSha':'0x01','XiangTan':'0x02',
        'ZhuZhou':'0x03','HengYang':'0x04',
        'ShaoYang':'0x05','YueYang':'0x06',
        'ZhangJiaJie':'0x07'}

# 硬件配置
class GUI_CONFIG:
    ESP32_URL = 'http://192.168.5.1:81/stream' # ESP32-CAM 流地址
    SERIAL_PORT = 'COM8'   # ################### 修改为你的串口号 ###################
    SERIAL_BAUDRATE = 9600 # 串口波特率
    TRIGGER_BYTES = b'\xaa\x55' # 触发识别的串口数据
    STOP_PREFIX = b'\xcc'       # 以 0xCC 开头的数据视为“光线弱”停止信号
    MODE_INBOUND_BYTES = b'\xaa\x44' # 切换到入库模式的指令
    MODE_OUTBOUND_BYTES = b'\xaa\x33'# 切换到出库模式的指令

# ===================================================================
# PyQt6 多线程工作器
# ===================================================================

class CameraThread(QThread):
    new_frame_signal = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._run_flag = True

    def run(self):
        cap = cv2.VideoCapture(GUI_CONFIG.ESP32_URL)
        if not cap.isOpened():
            print(f"错误: 无法打开视频流 {GUI_CONFIG.ESP32_URL}")
            return

        while self._run_flag:
            ret, frame = cap.read()
            if ret:
                self.new_frame_signal.emit(frame)
            self.msleep(30)
        
        cap.release()

    def stop(self):
        self._run_flag = False
        self.wait()

#SerialThread以处理模式切换指令
class SerialThread(QThread):
    trigger_signal = pyqtSignal()
    log_signal = pyqtSignal(str)
    stop_due_to_cc_signal = pyqtSignal()
    # 新增: 定义用于切换模式的信号
    mode_inbound_signal = pyqtSignal()
    mode_outbound_signal = pyqtSignal()

    def __init__(self, serial_port, parent=None):
        super().__init__(parent)
        self._run_flag = True
        self.ser = serial_port

    def run(self):
        if not self.ser or not self.ser.is_open:
            self.log_signal.emit("错误: 串口对象无效或未打开，监听线程退出。")
            return

        self.log_signal.emit(f"串口 {self.ser.port} 已连接，开始等待触发信号...")
        buf = bytearray()
        while self._run_flag:
            try:
                n = self.ser.in_waiting
                if n > 0:
                    chunk = self.ser.read(n)
                    if not chunk:
                        self.msleep(20)
                        continue
                    buf.extend(chunk)

                    # 检查是否有以 0xCC 开头的数据
                    if any(b == GUI_CONFIG.STOP_PREFIX[0] for b in buf):
                        self.log_signal.emit("检测到串口数据以 0xCC 开头，停止系统。")
                        self.stop_due_to_cc_signal.emit()
                        self.ser.reset_input_buffer()
                        buf.clear()
                        break

                    # 检查多个不同的指令
                    cmd_in = GUI_CONFIG.MODE_INBOUND_BYTES
                    cmd_out = GUI_CONFIG.MODE_OUTBOUND_BYTES
                    trig = GUI_CONFIG.TRIGGER_BYTES

                    # 优先检查模式切换指令，然后检查触发指令
                    idx_in = buf.find(cmd_in)
                    if idx_in != -1:
                        self.log_signal.emit(f"接收到入库模式切换指令: {cmd_in.hex(' ')}")
                        self.mode_inbound_signal.emit()
                        del buf[:idx_in + len(cmd_in)]
                        self.ser.reset_input_buffer()
                        continue # 处理完后继续下一次循环

                    idx_out = buf.find(cmd_out)
                    if idx_out != -1:
                        self.log_signal.emit(f"接收到出库模式切换指令: {cmd_out.hex(' ')}")
                        self.mode_outbound_signal.emit()
                        del buf[:idx_out + len(cmd_out)]
                        self.ser.reset_input_buffer()
                        continue # 处理完后继续下一次循环
                    
                    idx_trig = buf.find(trig)
                    if idx_trig != -1:
                        self.log_signal.emit(f"接收到触发信号: {trig.hex(' ')}")
                        self.trigger_signal.emit()
                        del buf[:idx_trig + len(trig)]
                        self.ser.reset_input_buffer()
                        continue # 处理完后继续下一次循环

                self.msleep(30)
            except serial.SerialException as e:
                self.log_signal.emit(f"串口读取错误: {e}")
                break
        
        self.log_signal.emit("串口监听线程已停止。")

    def stop(self):
        self._run_flag = False
        self.wait()

class WorkerSignals(QObject):
    finished = pyqtSignal(object)
    log = pyqtSignal(str)

class AnalysisWorker(QRunnable):
    def __init__(self, frame, yolo_model):
        super().__init__()
        self.frame = frame
        self.yolo_model = yolo_model
        self.signals = WorkerSignals()

    def run(self):
        self.signals.log.emit("开始分析当前帧...")
        result, status = run_analysis_on_frame(self.frame, self.yolo_model)
        self.signals.log.emit(f"处理状态: {status}")
        self.signals.finished.emit(result)

# ===================================================================
# 自定义对话框用于添加/编辑
# ===================================================================

class ItemDialog(QDialog):
    def __init__(self, parent=None, data=None):
        super().__init__(parent)
        self.setWindowTitle("货物信息")
        layout = QFormLayout()
        self.city_edit = QLineEdit(data[0] if data else "")
        self.id_edit = QLineEdit(data[1] if data else "")
        self.name_edit = QLineEdit(data[2] if data else "")
        layout.addRow("城市:", self.city_edit)
        layout.addRow("ID:", self.id_edit)
        layout.addRow("名称:", self.name_edit)
        buttons = QHBoxLayout()
        ok_btn = QPushButton("确定")
        cancel_btn = QPushButton("取消")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        buttons.addWidget(ok_btn)
        buttons.addWidget(cancel_btn)
        layout.addRow(buttons)
        self.setLayout(layout)

    def get_data(self):
        return self.city_edit.text(), self.id_edit.text(), self.name_edit.text()

# ===================================================================
# PyQt6 主窗口界面
# ===================================================================

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("智能物流分拣系统 (V3 - by DGZ)")
        self.setGeometry(100, 100, 1280, 720)

        self.current_cv_frame = None
        self.yolo_model = None
        self.row_counter = 0
        self.is_running = False
        self.serial_port = None

        self.mode_inbound = True

        self.camera_thread = None
        self.serial_thread = None
        self.thread_pool = QThreadPool()

        self.init_ui()
        
        self.log("正在加载YOLOv8模型...")
        self.yolo_model = load_yolo_model()
        if self.yolo_model:
            self.log("✅ YOLOv8模型加载成功！")
        else:
            self.log("❌ 错误: YOLO模型加载失败，请检查控制台输出。")
            self.btn_start.setEnabled(False)
            self.btn_start.setText("模型加载失败")

        self.apply_styles()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QHBoxLayout()
        main_widget.setLayout(main_layout)
        self.setCentralWidget(main_widget)

        # Left Panel
        left_panel = QVBoxLayout()
        self.btn_start = QPushButton("启动系统")
        self.btn_stop = QPushButton("停止系统")
        self.btn_start.setFixedHeight(40)
        self.btn_stop.setFixedHeight(40)
        self.btn_stop.setEnabled(False)
        left_panel.addWidget(self.btn_start)
        left_panel.addWidget(self.btn_stop)

        self.mode_group = QButtonGroup(self)
        self.rb_in = QRadioButton("入库")
        self.rb_out = QRadioButton("出库")
        self.rb_in.setChecked(True)
        self.rb_in.setVisible(False)
        self.rb_out.setVisible(False)
        self.mode_group.addButton(self.rb_in)
        self.mode_group.addButton(self.rb_out)
        left_panel.addWidget(self.rb_in)
        left_panel.addWidget(self.rb_out)
        left_panel.addStretch()

        # Center Panel
        center_panel = QVBoxLayout()
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["序号", "货物城市", "货物ID", "货物名称"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        center_panel.addWidget(self.table)

        bottom_buttons = QHBoxLayout()
        btn_add = QPushButton("添加")
        btn_edit = QPushButton("编辑")
        btn_delete = QPushButton("删除")
        btn_export = QPushButton("导出CSV")
        btn_add.clicked.connect(self.add_item)
        btn_edit.clicked.connect(self.edit_item)
        btn_delete.clicked.connect(self.delete_item)
        btn_export.clicked.connect(self.export_to_csv)
        bottom_buttons.addWidget(btn_add)
        bottom_buttons.addWidget(btn_edit)
        bottom_buttons.addWidget(btn_delete)
        bottom_buttons.addWidget(btn_export)
        center_panel.addLayout(bottom_buttons)

        # Right Panel
        right_panel = QVBoxLayout()
        self.camera_label = QLabel("摄像头画面将显示在此处")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        self.camera_label.setFixedSize(640, 480)
        self.log_browser = QTextBrowser()
        self.log_browser.setFont(QFont("Courier New", 10))
        right_panel.addWidget(self.camera_label)
        right_panel.addWidget(self.log_browser)

        main_layout.addLayout(left_panel, 1)
        main_layout.addLayout(center_panel, 4)
        main_layout.addLayout(right_panel, 3)

    def connect_signals(self):
        self.btn_start.clicked.connect(self.start_system)
        self.btn_stop.clicked.connect(self.stop_system)
        self.rb_in.toggled.connect(self.on_mode_change)

    # 连接新的信号和槽
    def connect_thread_signals(self):
        self.camera_thread.new_frame_signal.connect(self.update_camera_feed)
        self.serial_thread.trigger_signal.connect(self.handle_trigger)
        self.serial_thread.stop_due_to_cc_signal.connect(self.handle_stop_due_to_cc)
        self.serial_thread.log_signal.connect(self.log)
        self.serial_thread.mode_inbound_signal.connect(self.set_mode_inbound)
        self.serial_thread.mode_outbound_signal.connect(self.set_mode_outbound)

    def on_mode_change(self, checked):
        self.mode_inbound = checked
        mode_txt = "入库" if self.mode_inbound else "出库"
        self.log(f"模式切换为：{mode_txt}")

    def start_system(self):
        self.log("系统启动中...")
        
        try:
            self.serial_port = serial.Serial(GUI_CONFIG.SERIAL_PORT, GUI_CONFIG.SERIAL_BAUDRATE, timeout=0.1)
            self.log(f"✅ 串口 {GUI_CONFIG.SERIAL_PORT} 连接成功。")
        except serial.SerialException as e:
            self.log(f"❌ 错误: 无法打开串口 {GUI_CONFIG.SERIAL_PORT}: {e}")
            self.serial_port = None
            QMessageBox.critical(self, "启动失败", f"无法连接到串口 {GUI_CONFIG.SERIAL_PORT}。\n请检查设备连接和端口号设置。")
            return

        self.is_running = True
        self.camera_thread = CameraThread()
        self.serial_thread = SerialThread(self.serial_port)
        self.connect_thread_signals()
        self.camera_thread.start()
        self.serial_thread.start()
        
        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.rb_in.setVisible(True)
        self.rb_out.setVisible(True)
        self.log("✅ 系统已启动。")
    
    def stop_system(self):
        if not self.is_running:
            return
        self.log("系统停止中...")
        self.is_running = False
        
        if self.camera_thread and self.camera_thread.isRunning():
            self.camera_thread.stop()
        if self.serial_thread and self.serial_thread.isRunning():
            self.serial_thread.stop()
        
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
            self.log("串口已关闭。")
        self.serial_port = None

        self.camera_thread = None
        self.serial_thread = None
        
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.camera_label.setText("摄像头已关闭")
        self.camera_label.setStyleSheet("background-color: black; color: white;")
        self.log("⏹️ 系统已停止。")

    def handle_stop_due_to_cc(self):
        self.log("光线弱，请补光！")
        if self.is_running:
            self.stop_system()
        self.camera_label.setText("光线弱，请补光！")
        self.camera_label.setStyleSheet("background-color: black; color: white;")

    def update_camera_feed(self, cv_img):
        if not self.is_running:
            return
        self.current_cv_frame = cv_img
        qt_img = self.convert_cv_qt(cv_img)
        self.camera_label.setPixmap(qt_img)

    def handle_trigger(self):
        if not self.is_running or self.current_cv_frame is None:
            return
        self.log("⚡ 触发成功！正在将当前帧送去分析...")
        worker = AnalysisWorker(self.current_cv_frame.copy(), self.yolo_model)
        worker.signals.finished.connect(self.handle_analysis_result)
        worker.signals.log.connect(self.log)
        self.thread_pool.start(worker)

    def handle_analysis_result(self, result):
        if result is None:
            self.log("❌ 分析失败，未获取有效结果。")
            return
        if self.mode_inbound:
            self.process_inbound(result)
        else:
            self.process_outbound(result)
            
    # 响应串口指令的槽函数
    def set_mode_inbound(self):
        """响应串口指令，切换到入库模式"""
        # 只有在当前不是入库模式时才进行切换，避免重复操作和记录日志
        if not self.rb_in.isChecked():
            self.log("串口指令: 切换到入库模式。")
            self.rb_in.setChecked(True) # 通过操作UI控件来触发on_mode_change
    
    def set_mode_outbound(self):
        """响应串口指令，切换到出库模式"""
        # 只有在当前不是出库模式时才进行切换
        if not self.rb_out.isChecked():
            self.log("串口指令: 切换到出库模式。")
            self.rb_out.setChecked(True) # 通过操作UI控件来触发on_mode_change

    def parse_city_and_id(self, qr_data: str):
        if not qr_data or qr_data == 'N/A':
            return 'N/A', 'N/A'
        s = str(qr_data).strip()
        dash_idx = s.find('-')
        city = s[:dash_idx] if dash_idx > 0 else s
        item_id = s
        return city, item_id

    def id_exists_in_table(self, item_id: str) -> bool:
        rows = self.table.rowCount()
        for r in range(rows):
            cell = self.table.item(r, 2)
            if cell and cell.text() == item_id:
                return True
        return False

    def find_row_by_id(self, item_id: str):
        rows = self.table.rowCount()
        for r in range(rows):
            cell = self.table.item(r, 2)
            if cell and cell.text() == item_id:
                return r
        return -1
    
    def send_serial_confirmation(self, item_id: str):
        """
        解析ID，并向串口发送5字节的确认信号。
        格式: 0xBB + city_code(1B) + month(1B) + day(1B) + 0x00
        """
        if not self.serial_port or not self.serial_port.is_open:
            self.log("⚠️ 串口未连接，无法发送确认信号。")
            return
        
        try:
            parts = item_id.split('-')
            if len(parts) < 3:
                self.log(f"🔍 ID '{item_id}' 格式不正确，无法解析日期。")
                return
                
            city_name = parts[0]
            date_str = parts[2]

            if city_name not in dic_city:
                self.log(f"🔍 城市 '{city_name}' 不在字典中，无法发送信号。")
                return
                
            if len(date_str) != 4 or not date_str.isdigit():
                self.log(f"🔍 日期部分 '{date_str}' 格式不正确，应为4位数字MMDD。")
                return
                
            city_code = int(dic_city[city_name], 16)
            month = int(date_str[:2])
            day = int(date_str[2:])
            
            data_to_send = bytearray([0xBB, city_code, month, day, 0x00])
            
            self.serial_port.write(data_to_send)
            self.log(f"🚀 向串口发送数据: {data_to_send.hex(' ')}")
            
        except (ValueError, IndexError) as e:
            self.log(f"❌ 解析ID '{item_id}' 并发送串口信号时出错: {e}")
        except Exception as e:
            self.log(f"❌ 发送串口信号时发生未知错误: {e}")

    def process_inbound(self, result):
        qr_data = result.get('qr_code', 'N/A')
        detection = result.get('detection')

        item_name = "N/A"
        if detection:
            try:
                item_name = f"{detection.get('class_name','N/A')} ({float(detection.get('confidence',0.0)):.2f})"
            except Exception:
                item_name = str(detection)

        city, item_id = self.parse_city_and_id(qr_data)

        if item_id == 'N/A':
            self.log("❌ 未识别到有效二维码ID，入库失败。")
            return

        if self.id_exists_in_table(item_id):
            self.log("重复入库")
            return

        self.send_serial_confirmation(item_id)

        self.row_counter += 1
        row_position = self.table.rowCount()
        self.table.insertRow(row_position)
        self.table.setItem(row_position, 0, QTableWidgetItem(str(self.row_counter)))
        self.table.setItem(row_position, 1, QTableWidgetItem(city))
        self.table.setItem(row_position, 2, QTableWidgetItem(item_id))
        self.table.setItem(row_position, 3, QTableWidgetItem(item_name))
        self.log(f"✅ 入库成功：{item_id}")

    def process_outbound(self, result):
        qr_data = result.get('qr_code', 'N/A')
        city, item_id = self.parse_city_and_id(qr_data)

        if item_id == 'N/A':
            self.log("❌ 未识别到有效二维码ID，出库失败。")
            return

        row = self.find_row_by_id(item_id)
        if row == -1:
            self.log("出库失败：未找到该ID")
            return

        self.send_serial_confirmation(item_id)

        self.table.removeRow(row)
        self.log(f"✅ 出库成功：{item_id}")

    # 手动添加
    def add_item(self):
        dialog = ItemDialog(self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            city, item_id, name = dialog.get_data()
            if not item_id:
                QMessageBox.warning(self, "错误", "ID不能为空")
                return
            if self.id_exists_in_table(item_id):
                QMessageBox.warning(self, "错误", "ID已存在")
                return
            self.row_counter += 1
            row_position = self.table.rowCount()
            self.table.insertRow(row_position)
            self.table.setItem(row_position, 0, QTableWidgetItem(str(self.row_counter)))
            self.table.setItem(row_position, 1, QTableWidgetItem(city))
            self.table.setItem(row_position, 2, QTableWidgetItem(item_id))
            self.table.setItem(row_position, 3, QTableWidgetItem(name))
            self.log(f"✅ 手动添加成功：{item_id}")

    # 手动编辑
    def edit_item(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "错误", "请选择一行")
            return
        row = selected[0].row()
        data = [
            self.table.item(row, 1).text(),
            self.table.item(row, 2).text(),
            self.table.item(row, 3).text()
        ]
        dialog = ItemDialog(self, data)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            city, item_id, name = dialog.get_data()
            if not item_id:
                QMessageBox.warning(self, "错误", "ID不能为空")
                return
            old_id = data[1]
            if item_id != old_id and self.id_exists_in_table(item_id):
                QMessageBox.warning(self, "错误", "新ID已存在")
                return
            self.table.setItem(row, 1, QTableWidgetItem(city))
            self.table.setItem(row, 2, QTableWidgetItem(item_id))
            self.table.setItem(row, 3, QTableWidgetItem(name))
            self.log(f"✅ 手动编辑成功：{item_id}")

    # 手动删除
    def delete_item(self):
        selected = self.table.selectedItems()
        if not selected:
            QMessageBox.warning(self, "错误", "请选择一行")
            return
        row = selected[0].row()
        item_id = self.table.item(row, 2).text()
        self.table.removeRow(row)
        self.log(f"✅ 手动删除成功：{item_id}")

    # 导出到CSV
    def export_to_csv(self):
        path, _ = QFileDialog.getSaveFileName(self, "导出CSV", "", "CSV Files (*.csv)")
        if not path:
            return
        with open(path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["序号", "货物城市", "货物ID", "货物名称"])
            for row in range(self.table.rowCount()):
                writer.writerow([
                    self.table.item(row, 0).text(),
                    self.table.item(row, 1).text(),
                    self.table.item(row, 2).text(),
                    self.table.item(row, 3).text()
                ])
        self.log("✅ 数据导出成功")

    def log(self, message):
        self.log_browser.append(f"[{time.strftime('%H:%M:%S')}] {message}")

    def convert_cv_qt(self, cv_img):
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        p = qt_format.scaled(self.camera_label.width(), self.camera_label.height(), Qt.AspectRatioMode.KeepAspectRatio)
        return QPixmap.fromImage(p)



    def apply_styles(self):
        self.setStyleSheet("""
            QMainWindow { background-color: #1e1e1e; }
            QPushButton { background-color: #007acc; color: white; border: none; padding: 8px; border-radius: 4px; }
            QPushButton:hover { background-color: #005f99; }
            QTableWidget { background-color: #2d2d2d; color: white; gridline-color: #3c3c3c; selection-background-color: #007acc; }
            QHeaderView::section { background-color: #007acc; color: white; padding: 4px; border: none; }
            QTextBrowser { background-color: #2d2d2d; color: #00ff00; border: none; }
            QRadioButton { color: white; }
            QLabel { color: white; }
        """)
        palette = self.palette()
        gradient = QLinearGradient(0, 0, 0, self.height())
        gradient.setColorAt(0.0, QColor(30, 30, 30))
        gradient.setColorAt(1.0, QColor(50, 50, 50))
        palette.setBrush(QPalette.ColorRole.Window, QBrush(gradient))
        self.setPalette(palette)

    def closeEvent(self, event):
        self.stop_system()
        event.accept()

# ===================================================================
# 程序入口
# ===================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.connect_signals()
    main_win.show()
    sys.exit(app.exec())