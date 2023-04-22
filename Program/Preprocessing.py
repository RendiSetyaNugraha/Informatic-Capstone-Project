import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import sys

class Preprocessing:
    def __init__(self):
        pass

    #fungsi untuk membaca data citra
    def ReadImage(self, filename):
        try:
            img=cv2.imread(filename)
            return img
        except:
            print('Terjadi kesalahan pada proses pembacaan citra', sys.exc_info()[0])

    #fungsi untuk konversi citra RGB ke grayscale
    def RGB2Gray(self, img):
        try:
            imggray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            return imggray
        except:
            print('Terjadi kesalahan pada proses konversi citra RGB to Gray', sys.exc_info()[0])
    
    #fungsi untuk membaca data latih pada folder tertentu
    def ReadTrainingData(self,path,quality):
        try:
            #membaca detail lokasi dan nama citra
            imagesdata =[]
            imagesname=[]
            for r,d,f in os.walk(path):
                for img in f:
                    imagesdata.append(os.path.join(r, img))
                    imagesname.append(img)
            #mengetahui jumlah citra yang terdapat di dalam folder
            n=len(imagesdata)

            #menentukan jumlah baris dan kolom pada figure
            n_row = int(n/10)
            n_col = 10+(n%10)

            #menampilkan citra RGB data latih
            i=0
            plt.figure()
            for img in imagesdata:
                #membaca citra RGB
                image=self.ReadImage(img)
                b,g,r = cv2.split(image)
                image2=cv2.merge([r,g,b])

                #menampilkan citra RGB
                plt.subplot(n_row,n_col,i+1)
                plt.imshow(image2)
                i=i+1

            plt.suptitle('Green Apple (RGB) - Quality: '+quality)

            #menampilkan citra grayscale data latih
            i=0
            plt.figure()
            for img in imagesdata:
                #membaca dan mengkonversi citra RGB ke gray
                image=self.ReadImage(img)
                gray_image=self.RGB2Gray(image)

                #menampilkan citra grayscale
                plt.subplot(n_row, n_col,i+1)
                plt.imshow(gray_image, cmap=plt.get_cmap('gray'))
                i=i+1
            plt.suptitle('Green Apple (Grayscale) - Quality : '+quality)

            return[imagesdata,imagesname]
        except:
            print('error pada proses baca data latih', sys.exc_info()[0])
    
    #fungsi untuk membaca citra data uji dalam folder tertentu
    def ReadTestingData(self, path):
        try:
            imagesdata=[]
            imagesname=[]
            for r,d,f in os.walk(path):
                for img in f:
                    imagesdata.append(os.path.join(r, img))
                    imagesname.append(img)
            n=len(imagesdata)
            n_row = int(n/10)
            n_col = 10+(n%10)

            #menampilkan citra RGB data uji
            i=0
            plt.figure()
            for img in imagesdata:
                #membaca citra RGB
                image=self.ReadImage(img)
                b,g,r = cv2.split(image)
                image2=cv2.merge([r,g,b])

                #menampilkan citra RGB
                plt.subplot(n_row,n_col,i+1)
                plt.imshow(image2)
                i=i+1

            plt.suptitle('The testing data (rgb)')

            #menampilkan citra grayscale data uji
            i=0
            plt.figure()
            for img in imagesdata:
                #membaca dan mengkonversi citra RGB ke gray
                image=self.ReadImage(img)
                b,g,r = cv2.split(image)
                gray_image=self.RGB2Gray(image)

                #menampilkan citra grayscale
                plt.subplot(n_row, n_col,i+1)
                plt.imshow(gray_image, cmap=plt.get_cmap('gray'))
                i=i+1
            plt.suptitle('The Testing Data (grayscale)')

            return[imagesdata,imagesname]
        except:
            print('Error pada proses pembacaan data uji', sys.exc_info()[0])


    #fungsi untuk mengekstrasi fitur GLCM citra
    def GLCMFeatures(self, filename):
        try:
            image=self.ReadImage(filename)
            
            scale_percent = 10
            baris = int(image.shape[0] * scale_percent / 100)
            kolom = int(image.shape[1] * scale_percent / 75)
            dim = (kolom,baris)

            #resize image(upscale)
            resize = cv2.resize(image, dim, interpolation= cv2.INTER_AREA)
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
            
            Roi = cv2.cvtColor(Roi, cv2.COLOR_BGR2RGB)
#---------------------------------------------------------------------------------------------------
            gray_image=self.RGB2Gray(Roi)
            max_val=np.max(gray_image)

            #menciptakan matriks GLCM
            glcm=np.zeros((max_val+1, max_val+1), dtype=int)

            #menghitung frekuensi kemunculan matriks bertetangga
            row,col = gray_image.shape
            row1,col1=glcm.shape
            for i in range (row):
                for j in range(col-1):
                    window=gray_image[i, j:j+2]
                    glcm[window[0], window[1]] = glcm[window[0],window[1]]+1

            #membuat matriks GLCM simetris
            glcm_transpose = np.transpose(glcm)
            glcm=glcm+glcm_transpose

            #normalisasi matriks GCLM simetris
            n=np.sum(glcm)
            glcm_norm=glcm/n

            #menghitung fitur GLCM
            energy = 0
            contrast = 0
            entropy = 0
            homogeneity = 0
            row,col=glcm_norm.shape
            for i in range (row):
                for j in range (col):
                    energy = energy+np.power(glcm_norm[i,j],2)
                    contrast=contrast+(glcm_norm[i,j]*np.power((i-j), 2))
                    homogeneity=homogeneity+(glcm_norm[i,j]/(1+np.power((i-j),2)))

                    if glcm_norm[i,j] !=0:
                        entropy=entropy+(-np.log(glcm_norm[i,j])*glcm_norm[i,j])

            energy=round(energy,3)
            contrast=round(contrast,3)
            entropy=round(entropy,3)
            homogeneity=round(homogeneity,3)

            return [energy,contrast,entropy,homogeneity]
        except:
            print('Error pada ektrasi fitur GLCM', sys.exc_info()[0])

    #mendefinisikan fungsi untuk melakukan normalisasi data
    def Normalisasi(self, data):
        try:
            n_data = data.shape[0]
            x=np.zeros((n_data,1))
            datamax = max(data)
            datamin = min(data)
            for i in range (n_data):
                x[i, 0] = round((data[i]-datamin)/(datamax-datamin),3)

            return x
        except:
            print('Terjadi Kesalahan dapa normalisasi',sys.exc_info()[0])
    
    #mendefinisikan fungsi untuk melakukan denormalisasi
    def Denormalisasi(self, data, mindata, maxdata):
        try:
            x=round((data*maxdata-data*mindata)+mindata,3)

            return x
        except:
            print('Terjadi kesalahan pada fungsi denormalisasi', sys.exc_info()[0])
            
            
