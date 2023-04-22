import sys
import time
import cv2 as cv
import numpy as np
from pathlib import Path
import os, re, glob
import seaborn as sns
import pandas as pd
import shutil
import os

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from keras.optimizers import SGD, Adam

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import load_img, img_to_array

from sklearn.model_selection import train_test_split

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QMainWindow, QMessageBox, QTableWidgetItem
from PyQt5.QtWidgets import QDialog, QFileDialog
from PyQt5.QtGui import QIcon, QPixmap, QImage
from PyQt5.uic import loadUi
from tkinter import messagebox
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvas
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import *
# from ImageProcessing import *
from Preprocessing import *
# from cnn import *

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
                
                # Roi = cv.cvtColor(Roi, cv.COLOR_BGR2RGB)
                
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

        for i, path in enumerate([path_bagus, path_busuk]):
            for j, img_name in enumerate(os.listdir(path)[:5]): # Mengambil 5 gambar pertama dari setiap folder
                img_path = os.path.join(path, img_name)
                img = plt.imread(img_path)
                axes[i,j].imshow(img)
                axes[i,j].set_title(f'{path.split()[-1]}_{j+1}')
                axes[i,j].axis('off')

        plt.tight_layout()
        plt.show()

    def folderkos(self):
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
            
    def create_model(self):

        image_dir = Path('kosong')
        filepaths = list(image_dir.glob(r'**/*.jpg'))
        labels = list(map(lambda x: os.path.split(os.path.split(x)[0])[1], filepaths))

        filepaths = pd.Series(filepaths, name='Filepath').astype(str)
        labels = pd.Series(labels, name='Label')

        image_df = pd.concat([filepaths, labels], axis=1)
        # image_df

        train_df, test_df = train_test_split(image_df, train_size=0.7, shuffle=True, random_state=1)
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
            batch_size = 2,
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
            batch_size = 2,
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

        model_save_callback = callbacks.ModelCheckpoint('Percobaan_tkinter/model_{val_accuracy:.3f}_{accuracy:.3f}.h5', 
                                                            save_best_only=False, save_weights_only=False, monitor='val_accuracy')

        lrate = 0.0001
        adam = Adam(learning_rate=lrate)
        # sgd = SGD(learning_rate= lrate)
        model.compile(optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy'])

        model.fit(train_images,
                    validation_data=test_images,
                    epochs=50,
                    callbacks=[model_save_callback]
                    )
                    

        return model
                        
if __name__=="__main__":
    app = QApplication(sys.argv)
    form= klasifikasi()
    form.show()
    sys.exit(app.exec_())


