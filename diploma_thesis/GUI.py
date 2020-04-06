import pyqtgraph as pg
import random
from PyQt5.QtCore import QDateTime, Qt, QTimer
from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QButtonGroup, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QTextEdit)

class myApplication(QDialog):
    def __init__(self, parent = None):
        super(myApplication,self).__init__(parent)
        self.pallete = QApplication.palette()
        self.group_layout = None
        self.createControls()
        self.create_chart()

        self.mainLayout = QGridLayout()
        self.mainLayout.addWidget(self.controlGroup,0,0)
        self.mainLayout.addWidget(self.plotWidget,0,1)
        self.setLayout(self.mainLayout)
        self.resize(800,800)

        self.setWindowTitle('Diplomova praca Jakub Sarina')
        self.show()



    def createControls(self):
        self.controlGroup = QGroupBox('Controls')
        self.learn_rate = QLineEdit('0.001')
        self.learn_rate_label = QLabel('Learning rate:')
        self.num_o_ep = QLineEdit('20000')
        self.num_o_ep_label = QLabel('Number of episodes')
        self.gamma = QLineEdit('0.99')
        self.gamma_label = QLabel('Gamma:')
        self.egreedy = QLineEdit('0.99')
        self.egreedy_label = QLabel('Starting Egreedy:')
        self.decay = QLineEdit('0.99')
        self.decay_label = QLabel('Egreedy decay:')
        self.egreedy_final = QLineEdit('0.01')
        self.egreedy_final_label = QLabel('Final Egreedy:')
        self.score_to_achieve = QLineEdit('200')
        self.score_to_achieve_label = QLabel('Score to win:')
        self.report_interval = QLineEdit('10')
        self.report_interval_label = QLabel('Report interval:')
        self.number_o_hidden_layers = QLineEdit('2')
        self.number_o_layers_label = QLabel('Number of hidden layers:')
        self.hidden_layers = []
        self.number_o_hidden_layers.textChanged.connect(self.create_network_controls)
        self.memory_size = QLineEdit('600000')
        self.memory_size_label = QLabel('Replay memory size:')
        self.batch_size = QLineEdit('64')
        self.batch_size_label = QLabel('Batch size')
        self.update_target_frequency = QLineEdit('200')
        self.update_target_frequency_label = QLabel('Update target network frequency:')
        self.clip_err = QCheckBox('Error clipping enabled:')
        self.clip_err.setChecked(False)
        self.double_DQN = QRadioButton('Double DQN approach')
        self.double_DQN.setChecked(False)
        self.DQN = QRadioButton('DQN approach')
        self.DQN.setChecked(True)
        self.approach_group = QButtonGroup()
        self.approach_group.addButton(self.DQN)
        self.approach_group.addButton(self.double_DQN)
        self.dueling = QCheckBox('use Dueling NN architecture')
        self.dueling.setChecked(False)
        self.rendering_enabled = QCheckBox('Enable rendering')
        self.rendering_enabled.setChecked(False)
        self.rendering_enabled.stateChanged.connect(self.rendering_changed)
        self.load_weights_enabled = QCheckBox('Enable loading weights')
        self.load_weights_enabled.setChecked(False)
        self.load_weights_enabled.stateChanged.connect(self.load_weights_changed)
        self.save_weights_enabled = QCheckBox('Enable saving weights')
        self.save_weights_enabled.setChecked(False)
        self.save_weights_enabled.stateChanged.connect(self.add_to_chart)
        self.create_control_layout(True)

    def create_control_layout(self, first = False):



        if first == True:
            layout = QGridLayout()
            layout.addWidget(self.learn_rate_label, 0, 0)
            layout.addWidget(self.learn_rate, 0, 1)
            layout.addWidget(self.num_o_ep_label, 1, 0)
            layout.addWidget(self.num_o_ep, 1, 1)
            layout.addWidget(self.gamma_label, 2, 0)
            layout.addWidget(self.gamma, 2, 1)
            layout.addWidget(self.egreedy_label, 3, 0)
            layout.addWidget(self.egreedy, 3, 1)
            layout.addWidget(self.decay_label, 4, 0)
            layout.addWidget(self.decay, 4, 1)
            layout.addWidget(self.egreedy_final_label, 5, 0)
            layout.addWidget(self.egreedy_final, 5, 1)
            layout.addWidget(self.score_to_achieve_label, 6, 0)
            layout.addWidget(self.score_to_achieve, 6, 1)
            layout.addWidget(self.report_interval_label, 7, 0)
            layout.addWidget(self.report_interval, 7, 1)
            layout.addWidget(self.number_o_layers_label, 8, 0)
            layout.addWidget(self.number_o_hidden_layers, 8, 1)
            self.network_layout = QGridLayout()
            self.create_network_controls()
            layout.addLayout(self.network_layout, 9, 0, 1, -1)
            layout.addWidget(self.memory_size_label, 11, 0)
            layout.addWidget(self.memory_size, 11, 1)
            layout.addWidget(self.batch_size_label, 12, 0)
            layout.addWidget(self.batch_size, 12, 1)
            layout.addWidget(self.update_target_frequency_label, 13, 0)
            layout.addWidget(self.update_target_frequency, 13, 1)
            layout.addWidget(self.clip_err, 14, 0)
            layout.addWidget(self.DQN, 15, 0)
            layout.addWidget(self.double_DQN, 15, 1)
            layout.addWidget(self.dueling, 16, 0)
            layout.addWidget(self.rendering_enabled, 17, 0)
            layout.addWidget(self.save_weights_enabled, 19, 0)
            layout.addWidget(self.load_weights_enabled, 21, 0)
            self.save_layout = QGridLayout()
            self.rendering_layout = QGridLayout()
            self.load_layout = QGridLayout()
            layout.addLayout(self.save_layout, 20, 0, 1, -1)
            layout.addLayout(self.load_layout, 22, 0, 1, -1)
            layout.addLayout(self.rendering_layout, 18, 0, 1, -1)

            print('network created')
            self.group_layout = layout
            print(self.group_layout.count())
            self.controlGroup.setLayout(self.group_layout)
        else:
            pass

    def create_network_controls(self):
        if (self.number_o_hidden_layers.text() == '' or self.number_o_hidden_layers.text() is None):
            for i in range(self.network_layout.count()):
                tmp = self.network_layout.itemAt(0).widget()
                tmp.hide()
                self.network_layout.removeWidget(tmp)
                del tmp
            self.hidden_layers = []
            self.network_layout.update()
            return
        number = int(self.number_o_hidden_layers.text())
        print(number)
        hidden_layers = []
        for i in range(self.network_layout.count()):
            tmp = self.network_layout.itemAt(0).widget()
            tmp.hide()
            self.network_layout.removeWidget(tmp)
            del tmp
        for i in range(number):
            if len(self.hidden_layers) > 0:
                hidden_layers.append(QLineEdit(self.hidden_layers[i].text()))
            else:
                hidden_layers.append(QLineEdit())
            name = 'Hidden layer ' + (str(i+1)) + ': '
            self.network_layout.addWidget(QLabel(name),i,0)
            self.network_layout.addWidget(hidden_layers[i],i,1)
        self.hidden_layers = hidden_layers

    def rendering_changed(self):
        if self.rendering_enabled.isChecked() == True:
            self.render_frequency = QLineEdit('200')
            self.render_frequency_label = QLabel('Rendering frequency:')
            self.rendering_layout.addWidget(self.render_frequency_label, 0,0)
            self.rendering_layout.addWidget(self.render_frequency, 0,1)
        else:
            self.render_frequency.hide()
            self.render_frequency_label.hide()
            self.rendering_layout.removeWidget(self.rendering_layout.itemAt(0).widget())
            self.rendering_layout.removeWidget(self.rendering_layout.itemAt(0).widget())
            del self.render_frequency
            del self.render_frequency_label

    def load_weights_changed(self):
        if self.load_weights_enabled.isChecked() == True:
            self.load_weights_filename = QLineEdit()
            self.load_weights_filename_label = QLabel('Storage file:')
            self.load_layout.addWidget(self.load_weights_filename_label, 0, 0)
            self.load_layout.addWidget(self.load_weights_filename, 0, 1)
        else:
            self.load_weights_filename.hide()
            self.load_weights_filename_label.hide()
            self.load_layout.removeWidget(self.load_layout.itemAt(0).widget())
            self.load_layout.removeWidget(self.load_layout.itemAt(0).widget())
            del self.load_weights_filename
            del self.load_weights_filename_label


    def save_weights_changed(self, value):
        if self.save_weights_enabled.isChecked() == True:
            self.save_weights_filename = QLineEdit()
            self.save_weights_filename_label = QLabel('Storage file:')
            self.weight_saving_frequency = QLineEdit('100')
            self.weight_saving_frequency_label = QLabel('Weight saving frequency:')
            self.save_layout.addWidget(self.save_weights_filename_label,0,0)
            self.save_layout.addWidget(self.save_weights_filename,0,1)
            self.save_layout.addWidget(self.weight_saving_frequency_label,1,0)
            self.save_layout.addWidget(self.weight_saving_frequency,1,1)
        else:
            self.save_weights_filename.hide()
            self.save_weights_filename_label.hide()
            self.weight_saving_frequency.hide()
            self.weight_saving_frequency_label.hide()
            self.save_layout.removeWidget(self.save_layout.itemAt(0).widget())
            self.save_layout.removeWidget(self.save_layout.itemAt(0).widget())
            self.save_layout.removeWidget(self.save_layout.itemAt(0).widget())
            self.save_layout.removeWidget(self.save_layout.itemAt(0).widget())
            del self.save_weights_filename
            del self.save_weights_filename_label
            del self.weight_saving_frequency
            del self.weight_saving_frequency_label

    def create_chart(self):
        self.plotWidget = pg.PlotWidget()

        self.x = [1, 2, 3, 4, 5, 6]
        self.y = [200, 200, 250, 180, -5, 0]
        self.graph = self.plotWidget.plot(self.x, self.y)

    def add_to_chart(self):
        if len(self.x) < 30:
            self.x.append(self.x[-1] + 1)
            self.y.append(random.randint(0,50))
        else:
            self.x = self.x[1:]
            self.x.append(self.x[-1] + 1)
            self.y = self.y[1:]
            self.y.append(random.randint(0,50))
        self.graph.setData(self.x,self.y)


import sys
app = QApplication(sys.argv)
myApp = myApplication()
myApplication.show(myApp)
sys.exit(app.exec_())


