import pyqtgraph as pg
import random
from PyQt5.QtCore import QThread, pyqtSignal,Qt

from PyQt5.QtWidgets import (QApplication, QCheckBox, QComboBox, QDateTimeEdit,
        QDial, QDialog, QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
        QProgressBar, QPushButton, QRadioButton, QButtonGroup, QScrollBar, QSizePolicy,
        QSlider, QSpinBox, QStyleFactory, QTableWidget, QTabWidget, QWidget, QTextEdit)
from Agent import Agent
import threading as th
from multiprocessing import Pipe
import os

class myApplication(QDialog):
    def __init__(self, parent = None):
        super(myApplication,self).__init__(parent)
        self.pallete = QApplication.palette()
        self.group_layout = None
        self.create_controls()
        self.create_charts()
        self.tabs = QTabWidget()
        self.main_tab = QWidget()
        self.results_tab = QWidget()
        self.application_layout = QGridLayout()
        self.application_layout.addWidget(self.controlGroup,0,0)
        self.graphGroup = QGroupBox('Graphs')
        layout = QGridLayout()
        layout.addWidget(self.actual_score_chart,0,0)
        layout.addWidget(self.score_avg,0,1)
        layout.addWidget(self.score_avg_100,0,2)
        layout.addWidget(self.score_avg_report,1,0)
        layout.addWidget(self.steps_chart,1,1)
        layout.addWidget(self.egreedy_chart,1,2)
        self.graphGroup.setLayout(layout)
        self.graphGroup.setEnabled(False)
        self.application_layout.addWidget(self.graphGroup,0,1)
        self.create_progress_bar()
        self.main_tab.setLayout(self.application_layout)
        self.tabs.addTab(self.main_tab, 'Application')
        self.tabs.addTab(self.results_tab, 'Results displayer')
        self.main_layout = QGridLayout()
        self.main_layout.addWidget(self.tabs)
        self.setLayout(self.main_layout)
        self.setWindowTitle('Diplomova praca Jakub Sarina')
        self.resize(1900,960)
        self.show()


    def create_buttons(self):
        self.start = QPushButton('Start training session')
        self.stop = QPushButton('Stop')
        self.start.clicked.connect(self.on_click_start)
        self.stop.clicked.connect(self.on_click_stop)
        self.stop.setDisabled(True)
        self.button_layout = QGridLayout()
        self.button_layout.addWidget(self.stop,0,0)
        self.button_layout.addWidget(self.start,0,1)

    def on_click_start(self):
        config = {}
        config['learn_rate'] = float(self.learn_rate.text())
        config['num_o_ep'] = int(self.num_o_ep.text())
        config['gamma'] = float(self.gamma.text())
        config['egreedy'] = float(self.egreedy.text())
        config['decay'] = float(self.decay.text())
        config['egreedy_final'] = float(self.egreedy_final.text())
        config['score_to_achieve'] = int(self.score_to_achieve.text())
        config['report_interval'] = int(self.report_interval.text())
        config['number_o_hidden_layers'] = int(self.number_o_hidden_layers.text())
        config['hidden_layers'] = [int(i.text()) for i in self.hidden_layers]
        config['memory_size'] = int(self.memory_size.text())
        config['batch_size'] = int(self.batch_size.text())
        config['update_target_frequency'] = int(self.update_target_frequency.text())
        config['clip_err'] = self.clip_err.isChecked()
        config['double_DQN'] = self.double_DQN.isChecked()
        config['DQN'] = self.DQN.isChecked()
        config['dueling'] = self.dueling.isChecked()
        config['rendering_enabled'] = self.rendering_enabled.isChecked()
        config['load_weights_enabled'] = self.load_weights_enabled.isChecked()
        config['save_weights_enabled'] = self.save_weights_enabled.isChecked()
        config['variable_updating_enabled'] = self.variable_updating_enabled.isChecked()
        config['stats_output_file'] = self.save_data_filename.text()
        if self.load_weights_enabled.isChecked() == True:
            config['load_weights_filename'] = self.load_weights_filename.text()
        if self.rendering_enabled.isChecked() == True:
            config['render_frequency'] = int(self.render_frequency.text())
        if self.save_weights_enabled.isChecked() == True:
            config['weights_saving_frequency'] = int(self.weight_saving_frequency.text())
            config['save_weights_filename'] = self.save_weights_filename.text()
        if self.variable_updating_enabled.isChecked() == True:
            config['update_target_frequency_base'] = int(self.update_target_frequency_base.text())
            config['update_target_frequency_multiplicator'] = float(self.update_target_frequency_multiplicator.text())
            config['update_target_frequency_limit'] = int(self.update_target_frequency_limit.text())
        self.score_avg_report.setTitle('Average score last '+self.report_interval.text()+' episodes')
        self.stop.setDisabled(False)
        self.deamon = RefreshDeamon(config)
        self.deamon.signal.connect(self.callback)
        self.reset_progress_bar()
        self.reset_charts()
        self.progress_bar_layout.setEnabled(True)
        self.deamon.start()



    def on_click_stop(self):
        self.deamon.stop()
        self.stop.setDisabled(True)
        self.progress_bar_layout.setEnabled(False)

    def create_progress_bar(self):
        self.progress_bar = QProgressBar()
        self.progress_bar.setGeometry(0,0,1895,30)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat('% to win')
        self.progress_bar.setAlignment(Qt.AlignCenter)
        self.progress_bar_layout = QGridLayout()
        self.progress_bar_layout.addWidget(self.progress_bar, 0, 0, 1, -1)
        self.episode_label = QLabel()
        self.frames_label = QLabel()
        self.memory_level_label = QLabel()
        self.is_solved_label = QLabel()
        self.progress_bar_layout.addWidget(self.episode_label, 1,0)
        self.progress_bar_layout.addWidget(self.frames_label, 1,1)
        self.progress_bar_layout.addWidget(self.memory_level_label, 1,2)
        self.progress_bar_layout.addWidget(self.is_solved_label, 1,3)
        self.application_layout.addLayout(self.progress_bar_layout,1,0,1,-1)
        self.progress_bar_layout.setEnabled(False)

    def reset_progress_bar(self):
        self.episode_label.setText('Episode: 0/' + self.num_o_ep.text())
        self.frames_label.setText('Frames total: 0')
        self.memory_level_label.setText('Replay memory filling level: 0/' + self.memory_size.text())
        self.is_solved_label.setText('Solved: False')
        self.progress_bar.setMaximum(int(self.score_to_achieve.text()))
        self.progress_bar.setValue(0)

    def update_progress_bar_layout(self,episode,frames_total,memory_level,solved,score):
        self.episode_label.setText('Episode: {}/{}'.format(episode, self.num_o_ep.text()))
        self.frames_label.setText('Frames total: ' + frames_total)
        self.memory_level_label.setText('Replay memory filling level: {}/{}'.format(memory_level, self.memory_size.text()))
        self.is_solved_label.setText('Solved: ' + solved)
        self.progress_bar.setValue(float(score))

    def create_controls(self):
        self.controlGroup = QGroupBox('Controls')
        self.learn_rate = QLineEdit('0.001')
        self.learn_rate_label = QLabel('Learning rate:')
        self.num_o_ep = QLineEdit('20000')
        self.num_o_ep_label = QLabel('Number of episodes')
        self.gamma = QLineEdit('0.99')
        self.gamma_label = QLabel('Gamma:')
        self.egreedy = QLineEdit('0.99')
        self.egreedy_label = QLabel('Starting Egreedy:')
        self.decay = QLineEdit('0.999')
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
        self.variable_updating_enabled = QCheckBox('Enable variable target updating:')
        self.variable_updating_enabled.setChecked(False)
        self.variable_updating_enabled.stateChanged.connect(self.variable_updating_changed)
        self.rendering_enabled.stateChanged.connect(self.rendering_changed)
        self.load_weights_enabled = QCheckBox('Enable loading weights')
        self.load_weights_enabled.setChecked(False)
        self.load_weights_enabled.stateChanged.connect(self.load_weights_changed)
        self.save_weights_enabled = QCheckBox('Enable saving weights')
        self.save_weights_enabled.setChecked(False)
        self.save_data_filename_label = QLabel('Data storage filename')
        self.save_data_filename = QLineEdit('Data_output.txt')
        self.save_weights_enabled.stateChanged.connect(self.save_weights_changed)
        self.create_control_layout()

    def create_control_layout(self):
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
        layout.addWidget(self.save_data_filename_label, 14, 0)
        layout.addWidget(self.save_data_filename, 14, 1)
        layout.addWidget(self.clip_err, 15, 0)
        layout.addWidget(self.DQN, 16, 0)
        layout.addWidget(self.double_DQN, 16, 1)
        layout.addWidget(self.dueling, 17, 0)
        layout.addWidget(self.rendering_enabled, 18, 0)
        layout.addWidget(self.variable_updating_enabled, 20, 0)
        layout.addWidget(self.save_weights_enabled, 22, 0)
        layout.addWidget(self.load_weights_enabled, 24, 0)
        self.save_layout = QGridLayout()
        self.rendering_layout = QGridLayout()
        self.load_layout = QGridLayout()
        self.variable_updating_layout = QGridLayout()
        layout.addLayout(self.save_layout, 23, 0, 1, -1)
        layout.addLayout(self.load_layout, 25, 0, 1, -1)
        layout.addLayout(self.rendering_layout, 19, 0, 1, -1)
        layout.addLayout(self.variable_updating_layout, 21, 0, 1, -1)
        self.create_buttons()
        layout.addLayout(self.button_layout,26,0,1,-1)
        self.group_layout = layout
        self.controlGroup.setLayout(self.group_layout)


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

    def variable_updating_changed(self):
        if self.variable_updating_enabled.isChecked() == True:
            self.update_target_frequency_base = QLineEdit('50')
            self.update_target_frequency_base_label = QLabel('Update target frequency base:')
            self.update_target_frequency_multiplicator = QLineEdit('1.1')
            self.update_target_frequency_multiplicator_label = QLabel('Update target frequency multiplicator:')
            self.update_target_frequency_limit = QLineEdit('800')
            self.update_target_frequency_limit_label = QLabel('Update target frequency limit:')
            self.variable_updating_layout.addWidget(self.update_target_frequency_base_label,0,0)
            self.variable_updating_layout.addWidget(self.update_target_frequency_base,0,1)
            self.variable_updating_layout.addWidget(self.update_target_frequency_multiplicator_label,1,0)
            self.variable_updating_layout.addWidget(self.update_target_frequency_multiplicator,1,1)
            self.variable_updating_layout.addWidget(self.update_target_frequency_limit_label,2,0)
            self.variable_updating_layout.addWidget(self.update_target_frequency_limit,2,1)
        else:
            self.update_target_frequency_base.hide()
            self.update_target_frequency_base_label.hide()
            self.update_target_frequency_multiplicator.hide()
            self.update_target_frequency_multiplicator_label.hide()
            self.update_target_frequency_limit.hide()
            self.update_target_frequency_limit_label.hide()
            self.variable_updating_layout.removeWidget(self.variable_updating_layout.itemAt(0).widget())
            self.variable_updating_layout.removeWidget(self.variable_updating_layout.itemAt(0).widget())
            self.variable_updating_layout.removeWidget(self.variable_updating_layout.itemAt(0).widget())
            self.variable_updating_layout.removeWidget(self.variable_updating_layout.itemAt(0).widget())
            self.variable_updating_layout.removeWidget(self.variable_updating_layout.itemAt(0).widget())
            self.variable_updating_layout.removeWidget(self.variable_updating_layout.itemAt(0).widget())
            del self.update_target_frequency_base
            del self.update_target_frequency_base_label
            del self.update_target_frequency_multiplicator
            del self.update_target_frequency_multiplicator_label
            del self.update_target_frequency_limit
            del self.update_target_frequency_limit_label


    def create_charts(self):
        self.actual_score_chart = pg.plot()
        self.actual_score_chart.setTitle('Actual score')
        self.actual_score_chart.setLabel('left', 'score')
        self.actual_score_chart.setLabel('bottom', 'episode')
        self.actual_score_x = []
        self.actual_score_y = []
        self.actual_score_bar = pg.BarGraphItem(x=self.actual_score_x, height=self.actual_score_y, width=1, brush='b')
        self.actual_score_chart.addItem(self.actual_score_bar)
        self.steps_chart = pg.plot()
        self.steps_chart.setTitle('Steps per episode')
        self.steps_chart.setLabel('left', 'steps')
        self.steps_chart.setLabel('bottom', 'episode')
        self.steps_x = []
        self.steps_y = []
        self.steps_bar = pg.BarGraphItem(x=self.steps_x, height=self.steps_y, width=1, brush='b')
        self.steps_chart.addItem(self.steps_bar)
        self.score_avg = pg.PlotWidget()
        self.score_avg.setTitle('Average score')
        self.score_avg.setLabel('left', 'score')
        self.score_avg.setLabel('bottom', 'episode')
        self.score_avg_x = []
        self.score_avg_y = []
        self.score_avg_line = self.score_avg.plot(self.score_avg_x, self.score_avg_y)
        self.score_avg_100 = pg.PlotWidget()
        self.score_avg_100.setTitle('Average score last 100 episodes')
        self.score_avg_100.setLabel('left', 'score')
        self.score_avg_100.setLabel('bottom', 'episode')
        self.score_avg_100_x = []
        self.score_avg_100_y = []
        self.score_avg_100_line =self.score_avg_100.plot(self.score_avg_100_x, self.score_avg_100_y)
        self.score_avg_report = pg.PlotWidget()
        self.score_avg_report.setTitle('Average score last report interval episodes')
        self.score_avg_report.setLabel('left', 'score')
        self.score_avg_report.setLabel('bottom', 'episode')
        self.score_avg_report_x = []
        self.score_avg_report_y = []
        self.score_avg_report_line =self.score_avg_report.plot(self.score_avg_report_x, self.score_avg_report_y)
        self.egreedy_chart = pg.PlotWidget()
        self.egreedy_chart.setTitle('E-greedy')
        self.egreedy_chart.setLabel('left', 'egreedy')
        self.egreedy_chart.setLabel('bottom', 'episode')
        self.egreedy_chart_x = []
        self.egreedy_chart_y = []
        self.egreedy_chart_line =self.egreedy_chart.plot(self.egreedy_chart_x, self.egreedy_chart_y)

    def reset_charts(self):
        self.egreedy_chart_x = []
        self.egreedy_chart_y = []
        self.egreedy_chart_line.setData(self.egreedy_chart_x, self.egreedy_chart_y)
        self.steps_x = []
        self.steps_y = []
        self.steps_bar.setOpts(x= self.steps_x, height = self.steps_y)
        self.score_avg_x = []
        self.score_avg_y = []
        self.score_avg_line.setData(self.score_avg_x, self.score_avg_y)
        self.score_avg_100_x = []
        self.score_avg_100_y = []
        self.score_avg_100_line.setData(self.score_avg_100_x, self.score_avg_100_y)
        self.score_avg_report_x = []
        self.score_avg_report_y = []
        self.score_avg_report_line.setData(self.score_avg_report_x, self.score_avg_report_y)
        self.actual_score_x = []
        self.actual_score_y = []
        self.score_avg_report_line.setData(self.score_avg_report_x, self.score_avg_report_y)

    def update_egreedy(self, data):
        if len(self.egreedy_chart_x) == 0:
            self.egreedy_chart_x.append(0)
            self.egreedy_chart_y.append(data)
        elif len(self.egreedy_chart_x) < 50:
            self.egreedy_chart_x.append(self.egreedy_chart_x[-1] + 1)
            self.egreedy_chart_y.append(data)
        else:
            self.egreedy_chart_x = self.egreedy_chart_x[1:]
            self.egreedy_chart_y = self.egreedy_chart_y[1:]
            self.egreedy_chart_x.append(self.egreedy_chart_x[-1] + 1)
            self.egreedy_chart_y.append(data)
        self.egreedy_chart_line.setData(self.egreedy_chart_x, self.egreedy_chart_y)


    def update_steps(self,data):
        if len(self.steps_x) == 0:
            self.steps_x.append(0)
            self.steps_y.append(data)
        elif len(self.steps_x) < 50:
            self.steps_x.append(self.steps_x[-1] + 1)
            self.steps_y.append(data)
        else:
            self.steps_x = self.steps_x[1:]
            self.steps_y = self.steps_y[1:]
            self.steps_x.append(self.steps_x[-1] + 1)
            self.steps_y.append(data)
        self.steps_bar.setOpts(x= self.steps_x, height = self.steps_y)

    def update_score_avg(self, data):
        if len(self.score_avg_x) == 0:
            self.score_avg_x.append(0)
            self.score_avg_y.append(data)
        elif len(self.score_avg_x) < 50:
            self.score_avg_x.append(self.score_avg_x[-1] + 1)
            self.score_avg_y.append(data)
        else:
            self.score_avg_x = self.score_avg_x[1:]
            self.score_avg_y = self.score_avg_y[1:]
            self.score_avg_x.append(self.score_avg_x[-1] + 1)
            self.score_avg_y.append(data)
        self.score_avg_line.setData(self.score_avg_x, self.score_avg_y)

    def update_score_avg_100(self,data):
        if len(self.score_avg_100_x) == 0:
            self.score_avg_100_x.append(0)
            self.score_avg_100_y.append(data)
        elif len(self.score_avg_100_x) < 50:
            self.score_avg_100_x.append(self.score_avg_100_x[-1] + 1)
            self.score_avg_100_y.append(data)
        else:
            self.score_avg_100_x = self.score_avg_100_x[1:]
            self.score_avg_100_y = self.score_avg_100_y[1:]
            self.score_avg_100_x.append(self.score_avg_100_x[-1] + 1)
            self.score_avg_100_y.append(data)
        self.score_avg_100_line.setData(self.score_avg_100_x, self.score_avg_100_y)

    def update_score_avg_report(self, data):
        if len(self.score_avg_report_x) == 0:
            self.score_avg_report_x.append(0)
            self.score_avg_report_y.append(data)
        elif len(self.score_avg_report_x) < 50:
            self.score_avg_report_x.append(self.score_avg_report_x[-1] + 1)
            self.score_avg_report_y.append(data)
        else:
            self.score_avg_report_x = self.score_avg_report_x[1:]
            self.score_avg_report_y = self.score_avg_report_y[1:]
            self.score_avg_report_x.append(self.score_avg_report_x[-1] + 1)
            self.score_avg_report_y.append(data)
        self.score_avg_report_line.setData(self.score_avg_report_x, self.score_avg_report_y)

    def update_actual_score(self, data):
        if len(self.actual_score_x) == 0:
            self.actual_score_x.append(0)
            self.actual_score_y.append(data)
        elif len(self.actual_score_x) < 50:
            self.actual_score_x.append(self.actual_score_x[-1] + 1)
            self.actual_score_y.append(data)
        else:
            self.actual_score_x = self.actual_score_x[1:]
            self.actual_score_y = self.actual_score_y[1:]
            self.actual_score_x.append(self.actual_score_x[-1] + 1)
            self.actual_score_y.append(data)
        self.actual_score_bar.setOpts(x=self.actual_score_x, height=self.actual_score_y)

    def callback(self,data):
        parsed = data.split(',')
        if len(parsed) != 10:
            print('trash ' +  data)
        self.update_egreedy(float(parsed[0]))
        self.update_actual_score(float(parsed[1]))
        self.update_score_avg(float(parsed[2]))
        self.update_score_avg_100(float(parsed[3]))
        self.update_score_avg_report(float(parsed[4]))
        self.update_steps(int(parsed[5]))
        self.update_progress_bar_layout(parsed[6],parsed[7],parsed[8],parsed[9],parsed[3])


class RefreshDeamon(QThread):
    signal = pyqtSignal(object)

    def __init__(self, config):
        QThread.__init__(self)
        self.pipe_out, pipe_in = Pipe()
        self.agent = Agent(config, pipe_in)
        self.listening = False
        self.thread = th.Thread(target=self.agent.play,)

    def run(self):
        self.listening = True
        self.thread.start()
        while self.listening:
            data = self.pipe_out.recv()
            self.signal.emit(data)

    def stop(self):
        self.listening = False
        self.pipe_out.send('stop')
        self.thread.join()
        self.quit()

import sys
app = QApplication(sys.argv)
myApp = myApplication()
myApplication.show(myApp)
sys.exit(app.exec_())


