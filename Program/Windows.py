# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'Windows.ui'
#
# Created by: PyQt5 UI code generator 5.15.4
#
# WARNING: Any manual changes made to this file will be lost when pyuic5 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1240, 907)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Input_frame = QtWidgets.QFrame(self.centralwidget)
        self.Input_frame.setGeometry(QtCore.QRect(-10, 0, 331, 351))
        self.Input_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Input_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Input_frame.setObjectName("Input_frame")
        self.pushButton = QtWidgets.QPushButton(self.Input_frame)
        self.pushButton.setGeometry(QtCore.QRect(40, 300, 112, 34))
        self.pushButton.setObjectName("pushButton")
        self.layoutWidget = QtWidgets.QWidget(self.Input_frame)
        self.layoutWidget.setGeometry(QtCore.QRect(20, 20, 301, 271))
        self.layoutWidget.setObjectName("layoutWidget")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.layoutWidget)
        self.verticalLayout_3.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.inputfile_label = QtWidgets.QLabel(self.layoutWidget)
        self.inputfile_label.setObjectName("inputfile_label")
        self.horizontalLayout_2.addWidget(self.inputfile_label)
        self.inputfilepath_edit = QtWidgets.QLineEdit(self.layoutWidget)
        self.inputfilepath_edit.setObjectName("inputfilepath_edit")
        self.horizontalLayout_2.addWidget(self.inputfilepath_edit)
        self.inputfilepath_btn = QtWidgets.QToolButton(self.layoutWidget)
        self.inputfilepath_btn.setObjectName("inputfilepath_btn")
        self.horizontalLayout_2.addWidget(self.inputfilepath_btn)
        self.verticalLayout_3.addLayout(self.horizontalLayout_2)
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.inputfile_label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.inputfile_label_2.setObjectName("inputfile_label_2")
        self.horizontalLayout_5.addWidget(self.inputfile_label_2)
        self.mapfilepath_edit = QtWidgets.QLineEdit(self.layoutWidget)
        self.mapfilepath_edit.setObjectName("mapfilepath_edit")
        self.horizontalLayout_5.addWidget(self.mapfilepath_edit)
        self.inputfilepath_btn_2 = QtWidgets.QToolButton(self.layoutWidget)
        self.inputfilepath_btn_2.setObjectName("inputfilepath_btn_2")
        self.horizontalLayout_5.addWidget(self.inputfilepath_btn_2)
        self.verticalLayout_3.addLayout(self.horizontalLayout_5)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.learnrate_label = QtWidgets.QLabel(self.layoutWidget)
        self.learnrate_label.setObjectName("learnrate_label")
        self.horizontalLayout.addWidget(self.learnrate_label)
        self.learnrate_edit = QtWidgets.QLineEdit(self.layoutWidget)
        self.learnrate_edit.setObjectName("learnrate_edit")
        self.horizontalLayout.addWidget(self.learnrate_edit)
        self.verticalLayout_3.addLayout(self.horizontalLayout)
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.learnrate_label_2 = QtWidgets.QLabel(self.layoutWidget)
        self.learnrate_label_2.setObjectName("learnrate_label_2")
        self.horizontalLayout_3.addWidget(self.learnrate_label_2)
        self.K_edit = QtWidgets.QLineEdit(self.layoutWidget)
        self.K_edit.setObjectName("K_edit")
        self.horizontalLayout_3.addWidget(self.K_edit)
        self.verticalLayout_3.addLayout(self.horizontalLayout_3)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.max_n_label = QtWidgets.QLabel(self.layoutWidget)
        self.max_n_label.setObjectName("max_n_label")
        self.horizontalLayout_4.addWidget(self.max_n_label)
        self.max_n_edit = QtWidgets.QLineEdit(self.layoutWidget)
        self.max_n_edit.setObjectName("max_n_edit")
        self.horizontalLayout_4.addWidget(self.max_n_edit)
        self.verticalLayout_3.addLayout(self.horizontalLayout_4)
        self.pushButton_2 = QtWidgets.QPushButton(self.Input_frame)
        self.pushButton_2.setGeometry(QtCore.QRect(180, 300, 112, 34))
        self.pushButton_2.setObjectName("pushButton_2")
        self.result_frame = QtWidgets.QFrame(self.centralwidget)
        self.result_frame.setGeometry(QtCore.QRect(0, 350, 321, 561))
        self.result_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.result_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.result_frame.setObjectName("result_frame")
        self.scrollArea = QtWidgets.QScrollArea(self.result_frame)
        self.scrollArea.setGeometry(QtCore.QRect(0, 0, 321, 541))
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setObjectName("scrollArea")
        self.scrollAreaWidgetContents = QtWidgets.QWidget()
        self.scrollAreaWidgetContents.setGeometry(QtCore.QRect(0, 0, 319, 539))
        self.scrollAreaWidgetContents.setObjectName("scrollAreaWidgetContents")
        self.textBrowser = QtWidgets.QTextBrowser(self.scrollAreaWidgetContents)
        self.textBrowser.setGeometry(QtCore.QRect(0, 0, 321, 531))
        self.textBrowser.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.textBrowser.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarAlwaysOn)
        self.textBrowser.setObjectName("textBrowser")
        self.scrollArea.setWidget(self.scrollAreaWidgetContents)
        self.plot_frame = QtWidgets.QFrame(self.centralwidget)
        self.plot_frame.setGeometry(QtCore.QRect(320, 0, 1061, 921))
        self.plot_frame.setMaximumSize(QtCore.QSize(1061, 921))
        self.plot_frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.plot_frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.plot_frame.setObjectName("plot_frame")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.plot_frame)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(0, 0, 921, 881))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.plot_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.plot_layout.setSizeConstraint(QtWidgets.QLayout.SetDefaultConstraint)
        self.plot_layout.setContentsMargins(0, 0, 0, 0)
        self.plot_layout.setObjectName("plot_layout")
        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.pushButton.setText(_translate("MainWindow", "Train !!"))
        self.inputfile_label.setText(_translate("MainWindow", "輸入資料路徑"))
        self.inputfilepath_edit.setText(_translate("MainWindow", "./train4dAll.txt"))
        self.inputfilepath_btn.setText(_translate("MainWindow", "..."))
        self.inputfile_label_2.setText(_translate("MainWindow", "地圖資料路徑"))
        self.mapfilepath_edit.setText(_translate("MainWindow", "./軌道座標點.txt"))
        self.inputfilepath_btn_2.setText(_translate("MainWindow", "..."))
        self.learnrate_label.setText(_translate("MainWindow", "學習率"))
        self.learnrate_edit.setText(_translate("MainWindow", "0.01"))
        self.learnrate_label_2.setText(_translate("MainWindow", "K = "))
        self.K_edit.setText(_translate("MainWindow", "3"))
        self.max_n_label.setText(_translate("MainWindow", "迭代次數"))
        self.max_n_edit.setText(_translate("MainWindow", "100"))
        self.pushButton_2.setText(_translate("MainWindow", "Start!!!"))
