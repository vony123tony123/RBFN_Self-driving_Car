# -*- coding: utf-8 -*-
import time
import traceback

import matplotlib
from PyQt5.QtCore import pyqtSignal, QThread, QTimer

import numpy as np
from Windows import Ui_MainWindow
from rbfn import RBFN
from drawplot import drawPlot
import os
from toolkit import toolkit
# 导入程序运行必须模块
import sys
# PyQt5中使用的基本控件都在PyQt5.QtWidgets模块中
from PyQt5.QtWidgets import QApplication, QMainWindow, QFileDialog
matplotlib.use('Qt5Agg')

# 設定gui的功能
class MyMainWindow(QMainWindow, Ui_MainWindow):

    step=0#用來判斷要不要創建plotpicture

    def choosefileDialog(self):
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        filename, _ = QFileDialog.getOpenFileName(self, "QFileDialog.getOpenFileName()", "",
                                                  "All Files (*);;Text Files (*.txt)", options=options)
        if filename:
            self.inputfilepath_edit.setText(filename)
        else:
            self.inputfilepath_edit.setText("")

    def get_info(self):
        self.pushButton.setEnabled(False)
        try:
            trainFilePath = self.inputfilepath_edit.text()
            K = int(self.K_edit.text())
            learnrate = float(self.learnrate_edit.text())
            max_epochs = int(self.max_n_edit.text())

            rbfn = RBFN()
            dataset, answers = RBFN.readFile(trainFilePath)
            rbfn.train(dataset, answers, K, learnrate, max_epochs)
            self.dataset_dim = len(dataset[0])
            self.rbfn = rbfn

        except Exception:
            print(traceback.format_exc())
            pass

        self.pushButton.setEnabled(True)

    def startCar(self):
        mapFilePath = self.mapfilepath_edit.text()
        original_point, goal_points, self.boarder_points = drawPlot.readMapFile(mapFilePath)
        self.canva.drawMap(goal_points,self.boarder_points)

        self.save4DFile = open('track4D.txt', 'w')
        self.save6DFile = open('track6D.txt', 'w')

        self.currentPoint = original_point[:-1]
        self.currentPhi = original_point[-1]
        self.currentVector = np.array([100, 0])

        self.timer = QTimer()
        self.timer.setInterval(100)
        self.timer.timeout.connect(self.updatePlot)
        self.timer.start()

    def __init__(self, parent=None):
        super(MyMainWindow, self).__init__(parent)
        self.setupUi(self)
        self.inputfilepath_btn.clicked.connect(self.choosefileDialog)
        self.pushButton.clicked.connect(self.get_info)
        self.pushButton_2.clicked.connect(self.startCar)
        self.canva = drawPlot()
        self.plot_layout.addWidget(self.canva)

    def updatePlot(self):
        if self.currentPoint[1] < 37: 
            try:
                phi = self.currentPhi
                point = self.currentPoint
                self.canva.drawPoint(point)
                sensor_vectors = drawPlot.getSensorVector(self.currentVector, phi)
                sensor_distances = list()
                for sensor_vector in sensor_vectors:
                    cross_point = drawPlot.findCrossPoint(self.boarder_points, point, sensor_vector)
                    #self.canva.drawPoint(cross_point, 'r')
                    distance = toolkit.euclid_distance(cross_point, point)
                    sensor_distances.append(distance)
                sensor_distances = np.array(sensor_distances).flatten()
                if self.dataset_dim == 5:
                    data = np.concatenate([point, sensor_distances])
                else:
                    data = sensor_distances
                theta = self.rbfn.predict(data)
                # theta = answers[i]
                print('distance = ', sensor_distances)
                print('point = ', point)
                print('phi =', phi)
                print('theta = ', theta)
                print('---------------')
                self.save4DFile.write(' '.join(map(str, sensor_distances)) + ' '+str(theta)+'\n')
                self.save6DFile.write(' '.join(map(str, point))+ ' ' +' '.join(map(str, sensor_distances)) + ' '+str(theta)+'\n')
                if np.min(sensor_distances) < 3:
                    raise Exception("touch the wall of map")
                self.canva.updatePlot()
                self.currentPoint, self.currentPhi = drawPlot.findNextState(point, phi, theta)
            except Exception as e:
                print(e)
                traceback_output = traceback.format_exc()
                print(traceback_output)
                self.timer.stop()
                self.save4DFile.close()
                self.save6DFile.close()
        else:
            phi = self.currentPhi
            point = self.currentPoint
            self.canva.drawPoint(point)
            sensor_vectors = drawPlot.getSensorVector(self.currentVector, phi)
            sensor_distances = list()
            for sensor_vector in sensor_vectors:
                cross_point = drawPlot.findCrossPoint(self.boarder_points, point, sensor_vector)
                #self.canva.drawPoint(cross_point, 'r')
                distance = toolkit.euclid_distance(cross_point, point)
                sensor_distances.append(distance)
            sensor_distances = np.array(sensor_distances).flatten()
            if self.dataset_dim == 5:
                data = np.concatenate([point, sensor_distances])
            else:
                data = sensor_distances
            theta = self.rbfn.predict(data)
            # theta = answers[i]
            print('distance = ', sensor_distances)
            print('point = ', point)
            print('phi =', phi)
            print('theta = ', theta)
            print('---------------')
            self.save4DFile.write(' '.join(map(str, sensor_distances)) + ' '+str(theta)+'\n')
            self.save6DFile.write(' '.join(map(str, point))+ ' ' +' '.join(map(str, sensor_distances)) + ' '+str(theta)+'\n')
            self.canva.updatePlot()
            self.currentPoint, self.currentPhi = drawPlot.findNextState(point, phi, theta)


            self.timer.stop()
            self.save4DFile.close()
            self.save6DFile.close()


if __name__ == "__main__":
    # 固定的，PyQt5程序都需要QApplication对象。sys.argv是命令行参数列表，确保程序可以双击运行
    app = QApplication(sys.argv)
    # 初始化
    myWin = MyMainWindow()

    # 将窗口控件显示在屏幕上
    myWin.show()
    # 程序运行，sys.exit方法确保程序完整退出。
    sys.exit(app.exec_())

