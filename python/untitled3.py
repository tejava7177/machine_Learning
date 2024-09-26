# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 16:00:38 2024

@author: user
"""


import matplotlib.pyplot as plt
from sklearn import datasets


"""
digit=datasets.load_digits()

plt.figure(figsize=(5,5))
plt.imshow(digit.images[0], cmap=plt.cm.gray_r, interpolation='nearest')

plt.show()
print(digit.data[0])
print("이 숫자는 ",digit.target[0], "입니다.")
"""



lfw=datasets.fetch_lfw_people(min_faces_per_person=20, resize=0.4)

plt.figure(figsize=(50,90))

for i in range(8):
    plt.subplot(11,18,i+1)
    plt.imshow(lfw.images[i],cmap=plt.cm.bone)
    plt.title(lfw.target_names[lfw.target[i]])
    

plt.show()


"""
news=datasets.fetch_20newsgroups(subset='train')
print("******\n", news.data[3], "\n******")
print("이 문서의 부류는 <", news.target_names[news.target[0]],">입니다.")
"""