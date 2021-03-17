import numpy as np
from numpy import linalg as LA
import cv2

# Definitions
# window is defined by its top left corner point and x, y displacement
# e.g. window = [1, 1, 5, 7] is a window with top left corner at (1,1) and bottom right corner at (6,8) 

# Helper functions

def warp(image, p):
    # warp parameter p should be length 6, and be type float
    warp_matrix = p.reshape((2,3))
    rows, cols = image.shape
    return cv2.warpAffine(image, warp_matrix, (cols, rows))
    

def crop(image, window):
    return image[window[1]:window[1]+window[3]+1, window[0]:window[0]+window[2]+1]

def image_derivatives(img, kernel_size=3):
    Ix = cv2.Sobel(img,cv2.CV_64F,1,0,ksize=kernel_size)
    Iy = cv2.Sobel(img,cv2.CV_64F,0,1,ksize=kernel_size)
    return (Ix, Iy)
    
def get_jacobian(image):
    # for 2D
    rows, cols = image.shape
    x = np.arange(cols)
    y = np.arange(rows)
    x, y = np.meshgrid(x, y)
    zeros = np.zeros((rows, cols))
    ones = np.ones((rows, cols))
    row1 = np.stack((x, y, ones, zeros, zeros, zeros), axis=2)
    row2 = np.stack((zeros, zeros, zeros, x, y, ones), axis=2)
    return np.stack((row1, row2), axis=2)

# Main LK function

def LK_aligment(image, template, window):
    template = crop(template, window)
    image = crop(image, window)
    rows, cols = image.shape
    
    p = np.zeros(6)
    tol = 0.01
    itr = 0
    delta_p = np.inf
    
    while LA.norm(delta_p) > tol and itr < 100: 
        # 1. warp image with warp function
        warped_image = warp(image, p)
        
        # 2. subtract image from template to obtain error term
        error = template - warped_image
        
        # 3. compute gradient of image
        Ix, Iy = image_derivatives(image)
        gradient = np.stack((Ix, Iy), axis=2)
        gradient = np.expand_dims(gradient, axis=2)
        
        # 4. evaluate jacobian
        jacobian = get_jacobian(image)
        
        # 5. compute steepest descent
        steepest_descent = np.matmul(gradient, jacobian)
        
        # 6. compute inverse hessian
        steepest_descent_T = np.transpose(steepest_descent, (0,1,3,2))
        hessian = np.matmul(steepest_descent_T, steepest_descent)
        hessian_inv = LA.pinv(hessian).sum(axis=(0,1))
        
        # 7. compute delta p
        delta_p = np.matmul(steepest_descent_T, error.reshape((rows, cols, 1, 1))).sum(axis=(0,1))
        delta_p = np.dot(hessian_inv, delta_p).reshape(-1)
        
        # 8. update p
        p += delta_p
        itr += 1

"""
# test codes

import matplotlib.pyplot as plt
img_file = "drosophila.avi"
cap = cv2.VideoCapture(img_file)
ret,frame = cap.read()
cap.release()
frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
frame = warp(frame, np.array([1.0,0,100,0,1,100]))
plt.imshow(image_derivatives(frame)[1], cmap='Greys')

"""