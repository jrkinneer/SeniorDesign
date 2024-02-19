import numpy as np
import cv2

def hough_peaks(H, Q):
    """_summary_

    Args:
        img_edge (NDArray): array of a hough transform
        Q ('uint8'): the Q number of local maxima from the img you want to find

    Returns:
        NDArray: (Qx2) array of the indecis in img_edge of the local maxima
    """
    indices = np.argpartition(H.flatten(), -2)[-Q:]
    return np.vstack(np.unravel_index(indices, H.shape)).T

def hough_lines_acc(img_edges, theta=(-90,90)):
    """_summary_

    Args:
        img_edges (img): image as np array
        theta (int, int): tuple of the start and end point of range of theta, theta[0] must be less than theta[1]

    Returns:
        tuple: returns tuple of Hough cummulator array ('uint8'), theta and rho values for that array
    """
    #calculate diagonal size of the image, rho, and theta
    height, width = img_edges.shape
    diagonal = np.ceil(np.sqrt(height**2 + width**2))
    rhos = np.arange(-diagonal, diagonal + 1, 1)
    theta_vals = np.deg2rad(np.arange(theta[0], theta[1], 1))
    
    #initialize empty H accumulator array
    H = np.zeros((len(rhos), len(theta_vals)), dtype=np.uint64)
    
    #finds the edge indices in img_edges
    y_idx, x_idx = np.nonzero(img_edges)
    
    for i in range((len(x_idx))):
        x = x_idx[i]
        y = y_idx[i]
        for j in range(len(theta_vals)):
            rho = int((x * np.cos(theta_vals[j])) + (y * np.sin(theta_vals[j])) + diagonal)
            H[rho, j] += 1
    
    return (H.astype('uint8'), theta_vals, rhos)

def hough_lines_draw(img, peaks, rho, theta):
    temp = img.copy()
    for i in range(len(peaks)):
        r = rho[peaks[i][0]]
        t = theta[peaks[i][1]]
        a = np.cos(t)
        b = np.sin(t)
        x0 = a * r
        y0 = b * r
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        temp = cv2.line(temp, (x1,y1), (x2,y2), (255,0,0), 2)
        
    return temp

def hough_wrapper(input_img, Q=8):
    #make input grayscale
    black_and_white = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    
    #canny edge detection
    edges = cv2.Canny(black_and_white, np.min(black_and_white), np.max(black_and_white))
    
    #get accumulator array, thetas, and rho values
    h, theta, rho = hough_lines_acc(edges)
    
    #get Q peaks from array
    peaks = hough_peaks(h, Q)
    
    #draw visualized lines on the original image
    visualized = hough_lines_draw(input_img, peaks, rho, theta)
    
    return visualized
    
def hough_wrapper_cv2(input_img):
    black_and_white = cv2.cvtColor(input_img, cv2.COLOR_RGB2GRAY)
    
    #canny edge detection
    edges = cv2.Canny(black_and_white, np.min(black_and_white), np.max(black_and_white))
    
    rho = 1
    theta = np.pi/180
    threshold = 5
    min_line_length = 5
    max_line_gap = 1
    line_image = np.copy(input_img) #creating an image copy to draw lines on
    # Run Hough on the edge-detected image
    lines = cv2.HoughLinesP(edges, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)
    lines_2 = lines.copy()
    # Iterate over the output "lines" and draw lines on the image copy
    for line in lines:
        for x1,y1,x2,y2 in line:
            cv2.line(line_image,(x1,y1),(x2,y2),(255,255,255),2)
                
    return line_image, lines