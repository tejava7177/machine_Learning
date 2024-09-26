# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 13:23:46 2024

@author: user
"""

import matplotlib.pyplot as plt
import cv2

img = cv2.imread('dog.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

#show iage
plt.imshow(img)