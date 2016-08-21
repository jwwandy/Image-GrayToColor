from sklearn import svm
from sklearn import cluster
from sklearn.decomposition import PCA
from sklearn import preprocessing
import cv2
import numpy as np
import image
from parameters import *
from buildFeatures import *
from graphCut import *
from superColorPixel import *

def kmeans(img,K):
	kmeans = cluster.KMeans(K)
	img.labels = kmeans.fit_predict(img.kmeansfeatures)
	return kmeans.cluster_centers_

def svm_train(img):
	svm_clf = svm.LinearSVC(dual = False,class_weight = 'balanced')
	# img.features = img.labels.reshape((img.pixel_num,1))
	svm_clf.fit(img.features,img.labels)
	return svm_clf

def svm_predict(img,svm_clf):
	img.labels = svm_clf.predict(img.features)
	return svm_clf.decision_function(img.features)

def lab2color(test_img,centers):
	n = int((surfWindow - 1)/2)
	test_lab = np.zeros((test_img.H,test_img.W,3))
	H = test_img.H
	W = test_img.W
	for x in range(0,H - 2*n):
		for y in range(0,W - 2*n):
			test_lab[x+n,y+n,1:3] = centers[test_img.labels[(W-2*n)*x+y]]
	test_lab[:,:,0] = test_img.L
	return cv2.cvtColor(test_lab.astype(np.uint8),cv2.COLOR_Lab2BGR)

def lab2color2D(test_img,centers):
	H = test_img.H
	W = test_img.W
	test_lab = np.zeros((H,W,3))
	for x in range(0,H):
		for y in range(0,W):
			test_lab[x,y,1:3] = centers[test_img.labels[x*W+y]]
			# test_lab[x,y,1:3] = centers[test_img.labels[x,y]] 
	test_lab[:,:,0] = test_img.L
	test_lab = test_lab.astype(np.uint8)
	return cv2.cvtColor(test_lab,cv2.COLOR_Lab2BGR)

if __name__ == '__main__':
	#### read image
	train_img = image.Image('./images/mountain_color_resized.jpg')
	# cv2.imshow('training image',train_img.bgr)
	# cv2.waitKey(0)

	#### color discretization
	print ('Start Computing K-Means of K =',K)
	centers = kmeans(train_img,K)
	print ('End computing K-Means of K =',K)

	#### build feature space
	print ('Start building training features')
	pca_train = PCA(n_components = numFeatures ,whiten = True)
	min_max_scaler = preprocessing.MinMaxScaler()
	buildFeatureSpace(train_img,pca_train,min_max_scaler)
	print ('End building Training features')

	#### perform SVM training
	print ('Start Training image')
	svm_clf = svm_train(train_img)
	print ('End Training image')
	del train_img
	#### read test image
	test_img = image.Image('./images/mountain_gray_resized.jpg')
	# cv2.imshow('testing image',test_img.bgr)
	# cv2.waitKey(0)

	#### build test image features
	print ('Start bulding Test features')
	testImageFeatures(test_img,pca_train,min_max_scaler)

	#### SVM predict and get score
	print ('Start Predict labels for test image')
	score = -1 * svm_predict(test_img,svm_clf)
	print ('End Predict labels for test image')
	svm_img = lab2color(test_img,centers)
	cv2.imwrite('./house_svm_output.jpg',svm_img)
	#### perform graph cut
	# perfrom zero padding to score, since score is not estimated in boundaries
	print ('Start QPBO')
	score = score.reshape((int(test_img.H-2*n),int(test_img.W-2*n),K))
	unary = np.zeros((test_img.H, test_img.W, K))
	for i in range(K):
		unary[:,:,i] = np.lib.pad(score[:,:,i],(int(n),int(n)),'constant',constant_values=0)
	unary = unary.reshape((test_img.H*test_img.W,K))
	graphCut(test_img, centers, unary)
	print ('End QPBO')
	color_img = lab2color2D(test_img,centers)
	cv2.namedWindow('colorized image1',cv2.WINDOW_NORMAL)
	cv2.imshow('colorized image1',color_img)
	cv2.waitKey(0)
	cv2.imwrite('./result/house_graphcut_output.jpg',color_img)
	superColorPixel(test_img, centers)

	#### convert predicted label to image
	print ('Start Superpixel')
	color_img = lab2color2D(test_img,centers)
	print ('End Superpixel')

	#### show image
	print('Show colorized image')
	cv2.namedWindow('colorized image',cv2.WINDOW_NORMAL)
	cv2.imshow('colorized image',color_img)
	cv2.waitKey(0)

	cv2.imwrite('./result/house_superpixel_output.jpg',color_img)

	#### store colorized image
	# outFileName = './output/mountain_gray_resized/alpha%d.jpg' % alpha
	# cv2.imwrite(outFileName,color_img)

	cv2.destroyAllWindows()
