import cv2
import numpy as np
from parameters import *

class Image:
	def __init__(self,filename):
		self.bgr = cv2.imread(filename,1)
		self.Lab = cv2.cvtColor(self.bgr,cv2.COLOR_BGR2Lab)
		self.L = self.Lab[:,:,0]
		self.Lg1 = cv2.GaussianBlur(self.L, (0,0), blurSig1)
		self.Lg2 = cv2.GaussianBlur(self.L, (0,0), blurSig2)
		self.H, self.W = self.L.shape
		self.pixel_num = self.L.size
		H = int(self.H)
		W = int(self.W)
		k_feature_size = int((W-2*n)*(H-2*n))
		a = self.Lab[int(n):int(H-n),int(n):int(W-n),1].reshape(k_feature_size ,1)
		b = self.Lab[int(n):int(H-n),int(n):int(W-n),2].reshape(k_feature_size ,1)
		self.kmeansfeatures = np.concatenate((a,b),axis = 1)
		self.labels = []
		self.features = []
