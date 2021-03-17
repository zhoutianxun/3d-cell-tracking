import cv2
from LK_alignment_2D import LK_aligment
from goodfeatures import cornerDetection

def KLT(frame, next_frame, corners):
    # compute affine transformation for each corner between frame and next frame
    
    
    LK_aligment(next_frame, frame, window)   