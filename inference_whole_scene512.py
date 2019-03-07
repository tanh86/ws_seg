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
from utils import sliding_window, convert_to_rgb



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
        #print(pred.shape)
        #print(np.unique(pred))
        pred = Image.fromarray(pred.astype(np.uint8))
        pred = pred.convert('RGB')
        pred.save(filename[:-5]+"_"+ImList[i])
                
