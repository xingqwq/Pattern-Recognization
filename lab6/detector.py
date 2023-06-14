from PySide6.QtCore import QThread, Signal
import cv2 as cv
from openvino.runtime import Core
import numpy as np
import time
import utils
from tracker import faceItem, tracker, track
from ultralytics.yolo.utils import ROOT, yaml_load
from ultralytics.yolo.utils.checks import check_yaml

class detector(QThread):
    conSignal = Signal(np.ndarray, int, int, list)
    def __init__(self, cameraID = 0):
        super(detector, self).__init__()
        self.cameraID = 1
        self.grabStatus = 0
        self.id2class = yaml_load(check_yaml('data.yaml'))['names']
        self.cnt = 0
        self.timeTotal = 0
        self.lastTime = time.time()
        self.fps = 20
        self.faceTracker = tracker(30)
        # 实例化Core对象
        self.core = Core() 
        # 载入并编译模型
        self.model_onnx = self.core.read_model(model='./best_openvino_model/best.xml')
        self.net = self.core.compile_model(self.model_onnx, device_name="CPU")
        # 获得模型输出节点
        self.output_node = self.net.outputs[0]  # yolov8n只有一个输出节点
        self.ir = ir = self.net.create_infer_request()

    def getBlob(self, img):
        [height, width, _] = img.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = img
        scale = length / 640
        blob = cv.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640), swapRB=True)
        return blob, scale
    
    def decodeResult(self, outputs, rows):
        boxes = []
        scores = []
        class_ids = []
        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)
        result_boxes = cv.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)
        return result_boxes, boxes, class_ids, scores
    
    def drawBoxex(self, img, id, classID, confidence, x, y, x_plusW, y_plusH):
        label = f'{self.id2class[classID]} ({confidence:.2f})'
        color = utils.id2color(id)
        cv.rectangle(img, (x, y), (x_plusW, y_plusH), color, 2)
        cv.putText(img, label, (x - 10, y - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    def run(self):
        camera = cv.VideoCapture("/dev/video2")
        self.grabStatus = 1
        while self.grabStatus:
            ret, frame = camera.read()
            if not ret:
                print("[ERROR] 抓图图像失败")
                break
            # 将视频帧转换为灰度图像
            img = frame
            # 目标检测
            blob, scale = self.getBlob(img)
            outputs = self.ir.infer(blob)[self.output_node]
            outputs = np.array([cv.transpose(outputs[0])])
            result_boxes, boxes, class_ids, scores = self.decodeResult(outputs, outputs.shape[1])
            # 绘制结果
            faceList = []
            for i in range(len(result_boxes)):
                index = result_boxes[i]
                box = boxes[index]
                x = round(box[0] * scale)
                y = round(box[1] * scale)
                w = round((box[0] + box[2]) * scale)
                h = round((box[1] + box[3]) * scale)
                faceList.append(faceItem(x, y, w, h, class_ids[index], scores[index]))
            # 轨迹匹配
            self.faceTracker.apply(faceList)
            noMask = 0
            noMaskInfo = []
            for id in range(self.faceTracker.tracksNum):
                i = self.faceTracker.tracks[id]
                if i.status == 0:
                    continue
                if sum(i.maskStatus) >= 5:
                    noMaskInfo.append("脸部ID: {}\n当前是否佩戴口罩: {}\n置信度:{}\n\n".format(
                                id,
                                "已正确佩戴" if i.isMask == 0 else "未佩戴",
                                (i.maskStatus[0] / (sum(i.maskStatus)+1)) if i.isMask == 0 else  (i.maskStatus[1] / (sum(i.maskStatus)+1))
                            ))
                if i.isMask == 1:
                    noMask += 1
                if i.isUpdate == 1:
                    i.isUpdate = 0
                    lastFace = i.faceQueue[-1]
                    self.drawBoxex(img, id, i.isMask, lastFace.score, lastFace.x, lastFace.y, lastFace.w, lastFace.h)
            # 更新UI
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
            # 计算FPS
            if self.cnt == 0:
                self.lastTime = time.time()
                self.cnt += 1
            else:
                t = time.time()
                self.timeTotal += t - self.lastTime
                self.cnt += 1
                self.lastTime = t
            if self.cnt == 40:
                self.fps = 1/(self.timeTotal/(self.cnt-1))
                self.timeTotal = 0
                self.cnt = 0
            self.conSignal.emit(img, self.fps, noMask, noMaskInfo)
        # camera.release()
        # cv.destroyAllWindows()
    
    def closeGrab(self):
        self.grabStatus = 0