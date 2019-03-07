import numpy as np

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
            red[pred==label]=labels[label,0]
            green[pred==label]=labels[label,1]
            blue[pred==label]=labels[label,2]
        rgb = np.zeros((pred.shape[0], pred.shape[1], 3))
        rgb[:,:,0] = (red)
        rgb[:,:,1] = (green)
        rgb[:,:,2] = (blue)
        return rgb
