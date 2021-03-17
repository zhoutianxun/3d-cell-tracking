import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
import cv2

def image_derivatives(img, kernel_size=3):
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel_size)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel_size)
    return (Ix, Iy)

def cornerDetection(img, maxCorners, qualityLevel, minDistance):
    corners = []
    value = []
    
    row, col = img.shape
    img = cv2.GaussianBlur(img,(5,5),0)
    Ix, Iy = image_derivatives(img/256)
    
    window = np.ones((3,3))
    Ix2 = convolve(Ix**2, window)
    Iy2 = convolve(Iy**2, window)
    IxIy = convolve(Ix*Iy, window)
    
    R = np.stack((Ix2, IxIy, IxIy, Iy2)).T.reshape(row*col, 2,2)
    R = np.min(np.linalg.eigvals(R), axis=1).reshape((col, row)).T
    
    for i in range(0, row, minDistance):
        for j in range(0, col, minDistance):
            #print(R[i: i+minDistance, j: j+minDistance])
            if np.max(R[i: i+minDistance, j: j+minDistance]) > qualityLevel:
                idx = np.unravel_index(R[i: i+minDistance, j: j+minDistance].argmax(), R[i: i+minDistance, j: j+minDistance].shape)
                corners.append([idx[0]+i, idx[1]+j])
                value.append(R[idx[0]+i, idx[1]+j])

    maxCorners = min(len(value), maxCorners)
    return np.asarray(corners)[np.argpartition(value, -maxCorners)[-maxCorners:]] 


def test():
    img_file = "drosophila.avi"
    cap = cv2.VideoCapture(img_file)
    ret,frame = cap.read()
    
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,frame2 = cap.read()
    frame2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    
    corners = cornerDetection(frame, maxCorners = 50, qualityLevel = 0.1, minDistance = 20)
    implot = plt.imshow(frame, cmap='gray')
    plt.scatter(corners[:,1], corners[:,0], s= 2, marker=',', c='red')
    plt.show()

#test()