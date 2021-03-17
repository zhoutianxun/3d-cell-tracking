import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import correlate
import cv2

def image_derivatives(img, kernel_size=3):

    img = np.intc(img)
    
    k = np.array([[-1, 0, 1],
                  [-2, 0, 2],
                  [-1, 0, 1]])
    dx = np.array([k, k, k])
    dy = np.array([k.T, k.T, k.T])
    k = np.array([[-1, -1, 1],
                  [-1, -2, -1],
                  [-1, -1, 1]])
    dz = np.array([k, np.zeros((3,3)), -k])
    Ix = correlate(img, dx, mode='constant')
    Iy = correlate(img, dy, mode='constant')
    Iz = correlate(img, dz, mode='constant')

    return (Ix, Iy, Iz)

def cornerDetection(img, maxCorners, qualityLevel, minDistance):
    corners = []
    value = []
    
    depth, row, col = img.shape
    # gaussian blurring
    for k in range(depth):
        img[k,:,:] = cv2.GaussianBlur(img[k,:,:],(5,5),0)
    Ix, Iy, Iz = image_derivatives(img)
    
    window = np.ones((3,3,3))
    Ix2 = correlate(Ix**2, window)
    Iy2 = correlate(Iy**2, window)
    Iz2 = correlate(Iz**2, window)
    IxIy = correlate(Ix*Iy, window)
    IxIz = correlate(Ix*Iz, window)
    IyIz = correlate(Iy*Iz, window)
    
    R = np.stack((Ix2, IxIy, IxIz, IxIy, Iy2, IyIz, IxIz, IyIz, Iz2)).T.reshape(row*col*depth, 3,3)
    R = np.min(np.linalg.eigvals(R), axis=1).reshape((col, row, depth)).T
    
    for k in range(0, depth, minDistance):
        for i in range(0, row, minDistance):
            for j in range(0, col, minDistance):
                #print(R[i: i+minDistance, j: j+minDistance])
                if np.max(R[k: k+minDistance, i: i+minDistance, j: j+minDistance]) > qualityLevel:
                    idx = np.unravel_index(R[k: k+minDistance, i: i+minDistance, j: j+minDistance].argmax(), \
                                           R[k: k+minDistance, i: i+minDistance, j: j+minDistance].shape)
                    corners.append([idx[0]+k, idx[1]+i, idx[2]+j])
                    value.append(R[idx[0]+k, idx[1]+i, idx[2]+j])

    maxCorners = min(len(value), maxCorners)
    return np.asarray(corners)[np.argpartition(value, -maxCorners)[-maxCorners:]] 

def plot_3d(image):
    
    # hardcoded axes range, set according to image
    range_min = 0
    range_max = 550

    fig = plt.figure()
    ax = Axes3D(fig)
    ax.set_xlim3d(range_min,range_max)
    ax.set_ylim3d(range_min,range_max)
    ax.set_zlim3d(-(range_min+range_max)/2,(range_min+range_max)/2)
    
    x = np.arange(0, image.shape[2])
    y = np.arange(0, image.shape[1])
    xv, yv = np.meshgrid(x, y)
    
    threshold = 50
    for z in range(image.shape[0]): # size of z dimension
        points = np.argwhere(image[z,:,:] > threshold)
        colors = image[z,points[:,0],points[:,1]]
        ax.scatter(points[:,0], points[:,1], np.ones(points[:,0].shape)*z, marker=".", s=2, c=colors, cmap="Reds")
    #plt.show()

def test_3D():
    frame = np.load('3d_cell_image.npy')
    frame = frame[0,:,:,:]
    corners = cornerDetection(frame, maxCorners = 100, qualityLevel = 0.05, minDistance = 2)
    
    # hardcoded axes range, set according to image
    range_min = 0
    range_max = 550

    ax = plt.axes(projection ="3d")
    #ax.set_xlim3d(range_min,range_max)
    #ax.set_ylim3d(range_min,range_max)
    #ax.set_zlim3d(-(range_min+range_max)/2,(range_min+range_max)/2)
    
    ax.scatter3D(corners[:,2], corners[:,1], corners[:,0], marker='o', s=1, color = "green")
    
    x = np.arange(0, frame.shape[2])
    y = np.arange(0, frame.shape[1])
    xv, yv = np.meshgrid(x, y)
    
    threshold = 50
    for z in range(frame.shape[0]): # size of z dimension
        points = np.argwhere(frame[z,:,:] > threshold)
        colors = frame[z,points[:,0],points[:,1]]
        ax.scatter(points[:,1], points[:,0], np.ones(points[:,0].shape)*z, marker=".", s=2, c=colors, cmap="Reds")
    
    plt.show()

test_3D()