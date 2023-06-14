# This Python file uses the following encoding: utf-8
import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QInputDialog, QLabel, QWidget, QVBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QThread, Signal
import cv2 as cv
import threading
import time
from ui import Ui_MainWindow
from detector import detector

class MainWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.detectorPtr = detector(1)
        self.detectorPtr.conSignal.connect(self.setPic)
        self.ui.setupUi(self)
        self.detectorPtr.start()
        
    def setPic(self, img, fps, noMask, noMaskInfo):
        img = QImage(img, img.shape[1], img.shape[0], QImage.Format.Format_RGB888)
        self.ui.label_3.setPixmap(QPixmap.fromImage(img))
        self.ui.label_5.setText("检测帧率:{:.2f} FPS".format(fps))
        self.ui.label_6.setText(" {} ".format(noMask))
        # 设置未带口罩信息列表
        self.ui.textBrowser.clear()
        for i in noMaskInfo:
            self.ui.textBrowser.append(i)
        self.ui.textBrowser.update()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    ui = MainWindow()
    ui.show()
    # 关闭线程
    app.exec()
    ui.detectorPtr.closeGrab()
    time.sleep(5)