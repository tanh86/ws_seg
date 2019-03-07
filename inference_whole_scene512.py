import numpy as np
import pdb
import os
from FCN8 import build_full_fcn8
from keras import backend as K
from scipy import misc
import pickle
import glob
from PIL import Image
import scipy.ndimage as ndimage
import cv2

def sliding_window(image, stepSize, windowSize):
	for y in range(0, image.shape[0], stepSize):
		for x in range(0, image.shape[1], stepSize):
			yield (x, y, image[y:y + windowSize[1], x:x + windowSize[0]])

def convert_to_rgb(pred):
        Unlabelled = [0,0,0]
        Ear = [0, 255, 192]
        EarRegion = [204, 255, 204]
        labels = np.array([Unlabelled,EarRegion,Ear])
        red = pred.copy()
        green = pred.copy()
        blue = pred.copy()
        for label in range(0,3):
            r[pred==label]=labels[label,0]
            g[pred==label]=labels[label,1]
            b[pred==label]=labels[label,2]
        rgb = np.zeros((pred.shape[0], pred.shape[1], 3))
        rgb[:,:,0] = (red)
        rgb[:,:,1] = (green)
        rgb[:,:,2] = (blue)
        return rgb

fcn8 =  build_full_fcn8(512)

mean = np.array([97.097468964506632,96.166340987241611,54.261503589179839])
mean = np.asarray(mean,dtype=K.floatx())
std = np.array([62.340012,63.062275,63.062275])
std = np.asarray(std,dtype=K.floatx())

ImList = os.listdir("in/")

(winW, winH) = (512, 512)

for i in range(0,len(ImList)):
    X_val = ndimage.imread(os.path.join("in/",ImList[i]))
    X_val = cv2.copyMakeBorder(X_val,52,52,240,240,cv2.BORDER_CONSTANT,value=0)
    X_val = np.asarray(X_val,dtype=K.floatx())
    X_val -= mean
    X_val/=(std + K.epsilon())
    for filename in glob.glob(os.path.join("out", '*.hdf5')):
        fcn8.load_weights(filename)
        pred = np.zeros([2048, 3072, 2])
        image = X_val
        for (x, y, window) in sliding_window(image, stepSize=512, windowSize=(winW, winH)):
        	#print(window.shape[0])
        	#print(window.shape[1])
        	if window.shape[0] != winH or window.shape[1] != winW:
        		continue
        	window = np.expand_dims(window, axis=0)
        	y_p = fcn8.predict(window,verbose=1) 
        	pred[ y:y + winH,x:x + winW] = y_p[0].copy()
        pred = pred[52:1996,240:2832]
        
        pred = np.argmax(pred,axis=-1).reshape((1944, 2592))
        #print(pred.shape)
        pred = convert_to_rgb(pred)
        print(pred.shape)
        print(np.unique(pred))
        pred = Image.fromarray(pred.astype(np.uint8))
        pred = pred.convert('RGB')
        pred.save(filename[:-5]+"_"+ImList[i])
        