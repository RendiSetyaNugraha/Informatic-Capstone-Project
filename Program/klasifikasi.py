import sys
import time
import cv2 as cv
import numpy as np
from pathlib import Path
import os, re, glob
import seaborn as sns
import pandas as pd
import shutil
import random
import os

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.model_selection import train_test_split

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QMessageBox, QTableWidgetItem
from PyQt5.QtWidgets import QDialog, QFileDialog, QSplashScreen, QProgressBar, QPushButton, QFrame, QHBoxLayout, QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.uic import loadUi
from tkinter import messagebox
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPixmap, QMovie
from PyQt5.QtWidgets import *
# from ImageProcessing import *
from Preprocessing import *
# from cnn import *

class progresbar(QSplashScreen):
    def __init__(self):
        super(QSplashScreen, self).__init__()
        loadUi("splash.ui", self)
        self.setWindowFlag(Qt.FramelessWindowHint)
        pixmap = QPixmap("bg1.jpg")
        self.setPixmap(pixmap)

    def progress(self):
        for i in range(100):
            time.sleep(0.1)
            self.progressBar.setValue(i)

class klasifikasi(QMainWindow):
    ip=Preprocessing()
    # cnn=cnn()

    flag_trainingdata = 0
    flag_testingdata=0

    def __init__(self):
        QMainWindow.__init__(self)
        loadUi('apps.ui', self)
        self.setWindowTitle("IMPLEMETASI CNN UNTUK KLASIFIKASI KUALITAS BUAH APEL")
        self.btnBacaCitraLatih.clicked.connect(self.BacaCitraLatih)
        self.btnProcessing.clicked.connect(self.cropping)
        self.btnfolderkos.clicked.connect(self.folderkos)
        self.btnProsesPelatihan.clicked.connect(self.create_model)
        # self.btnHasil.clicked.connect(self.hasil)
        self.btnOpenFile.clicked.connect(self.pengujianManual)


        self.button = self.findChild(QPushButton, "btnOpenFile")
        self.label = self.findChild(QLabel, "label_3")

        self.show()


    #fungsi untuk membaca data citra latih
    def BacaCitraLatih(self):
        try:
            #open direktori dialog
            path=QFileDialog.getExistingDirectory (self, 'Silahkan pilih direktori citra latih berdasarkan kualitas buah',
                                                   '', 
                                                   QFileDialog.ShowDirsOnly|
                                                   QFileDialog.DontResolveSymlinks)
            quality=self.comboKualitasBuah.currentText()

            [imagesdata,imagesname] = self.ip.ReadTrainingData(path,quality)

            quality_code = ''
            if quality =='Apel bagus':
                quality_code='0'
            elif quality == 'Apel busuk':
                quality_code='1'

            #menampilkan data latih pada tabel
            n_trainingdata = self.tblDataLatih.rowCount()
            n=len(imagesdata)
            self.editTotalDataLatih.setText(str(n_trainingdata+n))
            self.tblDataLatih.setRowCount(n_trainingdata+n)

            
            for i in range(n) :
                #menampilkan data citra pada tabel data latih
                self.tblDataLatih.setItem((n_trainingdata+i),0,QTableWidgetItem(imagesname[i]))
                self.tblDataLatih.setItem((n_trainingdata+i),1,QTableWidgetItem(imagesdata[i]))
                self.tblDataLatih.setItem((n_trainingdata+i),2,QTableWidgetItem(quality))
                self.tblDataLatih.setItem((n_trainingdata+i),3,QTableWidgetItem(quality_code))

                #---------------- Cropping citra From Scratch ----------------#
                resizeimg = cv.imread(imagesdata[i], cv.IMREAD_COLOR)
                scale_percent = 10
                baris = int(resizeimg.shape[0] * scale_percent / 100)
                kolom = int(resizeimg.shape[1] * scale_percent / 75)
                dim = (kolom,baris)

                #resize image(upscale)
                resize = cv.resize(resizeimg, dim, interpolation= cv.INTER_AREA)
                # print('resize : ', resize.shape)

                #ukuran dari baris(y) dan coloum(x)
                row = resize.shape[0]
                col = resize.shape[1]

                #cari titik tengah
                xtengah = col // 2
                ytengah = row // 2
                titiktngh = (xtengah,ytengah)
                # print("titik tengah : ",titiktngh)

                #Dari titik tengah geser ke kanan sebesar 115pixel
                xcol = xtengah + 115
                ybar = ytengah - 91
                titikpst = (xcol,ybar)
                # print("titik pusat : ",titikpst)
                #mengambil are ROI
                Kiri  = xcol - 250
                Atas  = ybar 
                Kanan = xcol 
                Bawah = ybar + 210
                Roi = resize[Atas:Bawah, Kiri:Kanan]
                
                if quality == 'Apel bagus':
                    # Mengganti path tujuan folder kosong dengan direktori yang diinginkan
                    dest_folder = 'kosong/Apel bagus' 
                    # Menggabungkan nama file dengan path tujuan folder
                    dest_file = os.path.join(dest_folder, imagesname[i])
                    cv2.imwrite(dest_file, Roi)
                    # Mengkopi atau memindahkan file ke folder tujuan
                    # shutil.copy(imagesdata[i], dest_file) # atau shutil.move(imagesdata[i], dest_file) jika ingin memindahkan
                elif quality == 'Apel busuk':
                    dest_folder = 'kosong/Apel busuk' 
                    dest_file = os.path.join(dest_folder, imagesname[i])
                    cv2.imwrite(dest_file, Roi)
                    # shutil.copy(imagesdata[i], dest_file) # atau shutil.move(imagesdata[i], dest_file) jika ingin memindahkan
            plt.show()
       
        except:
            print('Terjadi error', sys.exc_info()[0])

    

    def cropping(self):
        fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(10, 5), sharex=True, sharey=True)
        path_bagus = 'kosong/Apel bagus'
        path_busuk = 'kosong/Apel busuk'

        images_bagus = os.listdir(path_bagus)
        images_busuk = os.listdir(path_busuk)

        random.shuffle(images_bagus)
        random.shuffle(images_busuk)

        for i, img_name in enumerate(images_bagus[:5]):
            img_path = os.path.join(path_bagus, img_name)
            img = plt.imread(img_path)
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'Apel bagus {i+1}')
            axes[0, i].axis('off')

        for i, img_name in enumerate(images_busuk[:5]):
            img_path = os.path.join(path_busuk, img_name)
            img = plt.imread(img_path)
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'Apel busuk {i+1}')
            axes[1, i].axis('off')

        plt.tight_layout()
        plt.show()

            
    def create_model(self):


        global train_images
        global test_images
        global history

        image_dir = Path('kosong')
        filepaths = list(image_dir.glob(r'**/*.png'))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        image_df = pd.concat([filepaths, labels], axis=1)
        # image_df

        train_df, test_df = train_test_split(image_df, train_size=0.8, shuffle=True, random_state=1)
        print(len(train_df))
        print(len(test_df))
        
        train_datagen = ImageDataGenerator(
                  rescale = 1./255,
                  rotation_range = 30,
                  horizontal_flip = True,
                  shear_range=0.3,
                  fill_mode = 'nearest',
                  width_shift_range = 0.2,
                  height_shift_range = 0.2,
                  zoom_range = 0.1,
        )

        test_datagen = ImageDataGenerator(
                            rescale = 1./255
        )

        train_images = train_datagen.flow_from_dataframe(  
            dataframe = train_df,
            x_col = 'Filepath',
            y_col = 'Label',
            target_size = (200, 200),
            # batch_size = 3,
            color_mode = "rgb",
            class_mode = "binary",
            shuffle = True,
            subset = 'training'
        )

        test_images = test_datagen.flow_from_dataframe( 
            dataframe = test_df,  
            x_col = 'Filepath',
            y_col = 'Label',
            target_size = (200, 200),
            # batch_size = 3,
            color_mode = "rgb",
            class_mode = "binary",
            shuffle = False,
        )

        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(200, 200, 3)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),

            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(200, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(500, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        model.summary()

        model_save_callback = callbacks.ModelCheckpoint('Percobaan_tkinter/model.h5', 
                                                            save_best_only=True, save_weights_only=False, monitor='val_accuracy')

        lrate = 0.0001
        adam = Adam(learning_rate=lrate)
        # sgd = SGD(learning_rate= lrate)
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])
        
        epoch = 10
        history = model.fit(train_images,
                    validation_data=test_images,
                    epochs=epoch,
                    batch_size = 10,
                    callbacks=[model_save_callback]
                    )
        
        # for i in range(epoch):
        #     time.sleep(0.1)
        #     progress = (i+1)*100/epoch
        #     self.barprogress_train.setValue(progress)

        #---------------- Hasil Proses Pelatihan ------------------#

        models.load_model('Percobaan_tkinter/model.h5')
        #train
        # score_CNN = loaded_model.evaluate(train_images, verbose=0)
        # self.editTrainingLoss.setText('{:.5f}'.format(score_CNN[0]))
        # self.editTrainingAkurasi.setText('{:.2f}%'.format(score_CNN[1]*100))
        # print('Train Loss: {:.5f}'.format(score_CNN[0]))
        # print('Train Accuracy: {:.2f}%'.format(score_CNN[1]*100))

        # #test
        # score_CNN = loaded_model.evaluate(test_images, verbose=0)
        # self.editValidationLoss.setText('{:.5f}'.format(score_CNN[0]))
        # self.editValidationAkurasi.setText('{:.2f}%'.format(score_CNN[1]*100))
        # print('Test Loss: {:.5f}'.format(score_CNN[0]))
        # print('Test Accuracy: {:.2f}%'.format(score_CNN[1]*100))

          
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']
        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs = range(len(acc))

        plt.figure(figsize=(2.28, 2.25))
        plt.plot(epochs, acc, 'r', label = 'Training Accuracy')
        plt.plot(epochs, val_acc, 'b', label = 'Validation Accuracy')
        plt.title('Accuracy')
        plt.legend()
        plt.savefig('Percobaan_tkinter/accuracy_graph.png', dpi = 100)
        plt.figure()

        plt.figure(figsize=(2.28, 2.25))
        plt.plot(epochs, loss, 'r', label = 'Training Loss')
        plt.plot(epochs, val_loss, 'b', label = 'Validation Loss')
        plt.title('Loss')
        plt.legend()
        plt.savefig('Percobaan_tkinter/loss.png', dpi = 100)
        # plt.show()

        pixmap = QPixmap('Percobaan_tkinter/accuracy_graph.png')
        self.lblGrafikKonversi.setPixmap(pixmap)

        pixmap2 = QPixmap('Percobaan_tkinter/loss.png')
        self.lblGrafikKonversi_2.setPixmap(pixmap2)

        self.editTrainingLoss.setText('{:.5f}'.format(loss[-1]))
        self.editTrainingAkurasi.setText('{:.2f}%'.format(acc[-1] * 100))

        self.editValidationLoss.setText('{:.5f}'.format(val_loss[-1]))
        self.editValidationAkurasi.setText('{:.2f}%'.format(val_acc[-1]*100))

    def folderkos(self):
        # menghapus tabel
        self.tblDataLatih.setRowCount(0)  # Mengatur jumlah baris tabel menjadi 0
        model = self.tblDataLatih.model()
        model.removeRows(0, model.rowCount())

        #menghapus total line edit
        self.editTotalDataLatih.clear()
        self.editTrainingLoss.clear()
        self.editTrainingAkurasi.clear()

        self.editValidationLoss.clear()
        self.editValidationAkurasi.clear()

        #menghapus grafik
        self.lblGrafikKonversi.clear()
        self.lblGrafikKonversi_2.clear()

        # Menghapus seluruh isi folder apel bagus
        folder_path = 'kosong/Apel bagus'
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.remove(file_path)
            except Exception as e:
                print('Error saat menghapus %s: %s' % (file_path, e))

        # Menghapus seluruh isi folder apel busuk
        folder_path = 'kosong/Apel busuk'
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.remove(file_path)
            except Exception as e:
                print('Error saat menghapus %s: %s' % (file_path, e))

        # Menghapus seluruh isi folder percobaan_tkinter
        folder_path = 'Percobaan_tkinter'
        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    os.remove(file_path)
            except Exception as e:
                print('Error saat menghapus %s: %s' % (file_path, e))

    def pengujianManual(self):
        
        fname = QFileDialog.getOpenFileName(self, "Open File", "D:\ICP_rendi", "All Files (*);;PNG Files (*.png);;Jpg Files (*.jpg)")
        
        #open the image
        self.pixmap = QPixmap(fname[0])
        self.label.setPixmap(self.pixmap)

        loaded_model = models.load_model('Percobaan_tkinter/model.h5')

        img_path = fname[0]

        img = load_img(img_path, target_size=(200, 200))

        do = cv.imread(img_path, cv.IMREAD_UNCHANGED)

        Open = np.ones((5,5))
        Close = np.ones((20))

        hsv = cv.cvtColor(do, cv.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])

        redmask1 = cv.inRange(hsv, lower_red, upper_red)

        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])

        redmask2 = cv.inRange(hsv, lower_red, upper_red)

        redmask = redmask1+redmask2
        maskOpen = cv.morphologyEx(redmask, cv.MORPH_OPEN, Open)
        maskClose = cv.morphologyEx(maskOpen, cv.MORPH_CLOSE, Close)

        maskFinal = maskClose

        cnt_r = 0
        for r in redmask:
            cnt_r = cnt_r+list(r).count(255)
        # print("merah", cnt_r)
        # cv.imshow('Red_Mask:', redmask)

        lower_green = np.array([50, 50, 50])
        upper_green = np.array([70, 255, 255])
        greenmask = cv.inRange(hsv, lower_green, upper_green)
        # cv.imshow('Green_mask:', greenmask)
        cnt_g = 0
        for g in greenmask:
            cnt_g = cnt_g+list(g).count(255)
        # print("Hijau ", cnt_g)

        lower_yellow = np.array([20, 50, 50])
        upper_yellow = np.array([30, 255, 255])
        yellowmask = cv.inRange(hsv, lower_yellow, upper_yellow)
        # cv.imshow('Yellow Mask:', yellowmask)
        cnt_y = 0
        for y in yellowmask:
            cnt_y = cnt_y+list(y).count(255)
        # print("Kuning ", cnt_y)

        tot_area = cnt_r+cnt_y+cnt_g
        rperc = cnt_r/tot_area
        yperc = cnt_y/tot_area
        gperc = cnt_g/tot_area

        glimit = 0.5
        ylimit = 0.8

        if gperc>glimit: #segar
            #prediksi
            imgplot = plt.imshow(img)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis = 0)

            images = np.vstack([x])
            prediction = loaded_model.predict(images)
            print(prediction[0])
            if prediction[0] < 0.5:
                self.lblprediksi.setText("Apel Bagus")
            else:
                self.lblprediksi.setText("Apel Busuk")

        elif yperc>glimit: #busuk
            imgplot = plt.imshow(img)
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis = 0)

            images = np.vstack([x])
            prediction = loaded_model.predict(images)
            print(prediction[0])
            if prediction[0] < 0.5:
                self.lblprediksi.setText("Apel Bagus")
            else:
                self.lblprediksi.setText("Apel Busuk")

        else: #bukan buah apel
            self.lblprediksi.setText("Ini bukan buah apel hijau")
        
                        
if __name__=="__main__":
    app = QApplication(sys.argv)

    splash = progresbar()
    splash.show()
    splash.progress()

    form= klasifikasi()
    form.show()

    splash.finish(form)
    sys.exit(app.exec_())


