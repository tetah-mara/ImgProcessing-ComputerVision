# %% numpy
"""
- matrisler için hesaplama kolaylığı sağlar
"""
import numpy as np

# 1*15 boyutunda bir array-dizi
dizi = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])
print(dizi)

print(dizi.shape) # array'in boyutu

dizi2 = dizi.reshape(3, 5)

print("Şekil: ",dizi2.shape)
print("Boyut: ",dizi2.ndim)
print("Veri tipi: ",dizi2.dtype.name)
print("Boy: ",dizi2.size) 

# array type
print("Type: ",type(dizi2))

# 2 boyutlu array
dizi2D = np.array([[1,2,3,4],[5,6,7,8],[9,8,7,5]])
print(dizi2D)

# sıfırlardan oluşan bir array
sifir_dizi = np.zeros((3,4))
print(sifir_dizi)

# birlerden oluşan bir array
bir_dizi = np.ones((3,4))
print(bir_dizi)

# bos array
bos_dizi = np.empty((3,4))
print(bos_dizi)

# arange(x,y,basamak)
dizi_aralik = np.arange(10,50,5)
print(dizi_aralik)

# linspace(x,y, basamak) 
dizi_bosluk = np.linspace(10,20,5)
print(dizi_bosluk)

# float array
float_array = np.float32([[1,2],[3,4]])
print(float_array)

# matematiksel işlemler
a = np.array([1,2,3])
b = np.array([4,5,6])

print(a+b)
print(a-b)
print(a**2)

# dizi elemanı toplama
print(np.sum(a))

# max değer
print(np.max(a))

# min değer
print(np.min(a))

# mean ortalama
print(np.mean(a))

# median ortalama
print(np.median(a))

# rastgele sayı üretme [0,1] arasında sürekli uniform 3*3
rastgele_dizi = np.random.random((3,3))
print(rastgele_dizi)

# indeks
dizi = np.array([1,2,3,4,5,6,7])
print(dizi[0])

# dizinin ilk 4 elemanı
print(dizi[0:4])

# dizinin tersi
print(dizi[::-1])

# 
dizi2D = np.array([[1,2,3,4,5],[6,7,8,9,10]])
print(dizi2D)

# dizinin 1. satır ve 1. sutununda bulunan elemanı
print(dizi2D[1,1])

# 1. sütun ve tüm satılar
print(dizi2D[:,1])

# satır 1, sütun 1,2,3
print(dizi2D[1,1:4])

# dizinin son satır tüm sütunları
print(dizi2D[-1, :])

# 
dizi2D = np.array([[1,2,3],[4,5,6],[7,8,9]])
print(dizi2D)

# vektör haline getirme
vektor = dizi2D.ravel()
print(vektor)

maksimum_sayinin_indeksini = vektor.argmax()
print(maksimum_sayinin_indeksini)

# %% pandas
"""
- hızlı güçlü ve esnek istatiksel bilgi almanızı sağlar.
"""

import pandas as pd

# sözlük oluştur
dictionary = {"isim" : ["ali","veli","kenan","murat","ayse","hilal"],
              "yas"  : [15,16,17,33,45,66],
              "maas" : [100,150,240,350,110,220]}

veri = pd.DataFrame(dictionary)
print(veri)

# ilk 5 satır
print(veri.head())
print(veri.columns)
# veri bilgisi
print(veri.info())

# istatistiksel özellikler
print(veri.describe())

# yas sütunu
print(veri["yas"])

# sütun eklemek
veri["sehir"] = ["Ankara","İstanbul","Konya","İzmir","Bursa","Antalya"]
print(veri)

# yas sütunu
print(veri.loc[:,"yas"])

# yas sütunu ve 3 satır
print(veri.loc[:2,"yas"])

# yas ve şehir arası sütunu ve 3 satır
print(veri.loc[:2,"yas":"sehir"])

# yas ve şehir arası sütunu ve 3 satır
print(veri.loc[:2,["yas","isim"]])

# satırları tersten yazdır.
print(veri.loc[::-1,:])

# yas sütunu with iloc
print(veri.iloc[:,1])

# ilk 3 satır ve yaş ve isim
print(veri.iloc[:3,[0,1]])

# filtreleme
dictionary = {"isim" : ["ali","veli","kenan","murat","ayse","hilal"],
              "yas"  : [15,16,17,33,45,66],
              "sehir": ["İzmir","Ankara","Konya","Ankara","Ankara","Antalya"]}

veri = pd.DataFrame(dictionary)
print(veri)

# ilk olarak yaşa göre bir filtre yas > 22
filtre1 = veri.yas > 22
filtrelenmis_veri = veri[filtre1]
print(filtrelenmis_veri)

# ortalama yas 
ortalama_yas = veri.yas.mean()

veri["YAS_GRUBU"] = ["kucuk" if ortalama_yas > i else "buyuk" for i in veri.yas]
print(veri)

# birleştirme
sozluk1 = {"isim": ["ali","veli","kenan"],
              "yas" : [15,16,17],
              "sehir": ["İzmir","Ankara","Konya"]} 
veri1 = pd.DataFrame(sozluk1)

# veri seti 2 oluşturalım
sozluk2 = {"isim": ["murat","ayse","hilal"],
              "yas" : [33,45,66],
              "sehir": ["Ankara","Ankara","Antalya"]} 
veri2 = pd.DataFrame(sozluk2)

# dikey
veri_dikey = pd.concat([veri1, veri2], axis = 0)

# yatay
veri_yatay = pd.concat([veri1, veri2], axis = 1)


# %% matplotlib
"""
- görselleştirme
"""
import matplotlib.pyplot as plt
import numpy as np

x = np.array([1,2,3,4])
y = np.array([4,3,2,1])
plt.figure()
plt.plot(x,y, color="red",alpha = 0.7, label = "line")
plt.scatter(x,y,color = "blue",alpha= 0.4, label = "scatter")
plt.title("Matplotlib")
plt.xlabel("x")
plt.ylabel("y")
plt.grid(True)
plt.xticks([0,1,2,3,4,5])
plt.legend()
plt.show()

fig, axes = plt.subplots(2,1, figsize=(9,7))
fig.subplots_adjust(hspace = 0.5)

x = [1,2,3,4,5,6,7,8,9,10]
y = [10,9,8,7,6,5,4,3,2,1]


axes[0].scatter(x,y)
axes[0].set_title("sub-1")
axes[0].set_ylabel("sub-1 y")
axes[0].set_xlabel("sub-1 x")

axes[1].scatter(y,x)
axes[1].set_title("sub-2")
axes[1].set_ylabel("sub-2 y")
axes[1].set_xlabel("sub-2 x")

# random resim
plt.figure()
img = np.random.random((50,50))
plt.imshow(img, cmap = "gray") # 0(siyah) 1(beyaz) -> 0.5(gri) 
plt.show()

# %% OS
import os

print(os.name)

currentDir = os.getcwd()
print(currentDir)

# new folder
folder_name = "new_folder"
os.mkdir(folder_name)

new_folder_name = "new_folder_2"
os.rename(folder_name, new_folder_name)

os.chdir(currentDir+"\\"+new_folder_name)
print(os.getcwd())

os.chdir(currentDir)
print(os.getcwd())

files = os.listdir()

for f in files:
    if f.endswith(".py"):
        print(f)

os.rmdir(new_folder_name)

for i in os.walk(currentDir):
    print(i)
