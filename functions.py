import matplotlib.pyplot as plt
import numpy as np
import cv2
import skimage as ski
import numpy.fft as fft
import scipy.signal as signal

#General Lane Detection Code
def canny_out(image, sigma_g=15,dil_window=2,l_thresh=200,u_thresh=150):
    
    #Convert image to gray before edge filtering
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Apply gaussian filtering
    blurred_img = cv2.GaussianBlur(gray_image, (sigma_g, sigma_g), 0)
    # _, otsu_thresh_image = cv2.threshold(blurred_img, 0, 255, cv2.THRESH_OTSU)
    # neg_otsu_thresh_image = otsu_thresh_image
    kernel = np.ones((dil_window, dil_window), np.uint8)  # You can adjust the kernel size for more or less dilation
    # dilated_image = cv2.dilate(neg_otsu_thresh_image, kernel, iterations=1)
    cannyed_image = cv2.Canny(blurred_img, 2,20)
    # cannyed_image = cv2.bitwise_and(cannyed_image,dilated_image)

    return gray_image, cannyed_image

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    match_mask_color = 255
    cv2.fillPoly(mask, vertices, match_mask_color)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    line_img = np.zeros((img.shape[0],img.shape[1],3),dtype=np.uint8)
    img = np.copy(img)
    if lines is None:
        return
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)
    return img
def get_line_image(image,lines,thickness=30,T1=0.4,T2=0.999):

    left_line_x = []
    left_line_y = []
    right_line_x = []
    right_line_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope)<T1 or abs(slope)>T2:
                continue

            if slope <= 0:
                left_line_x.extend([x1, x2])
                left_line_y.extend([y1, y2])
            else:
                right_line_x.extend([x1, x2])
                right_line_y.extend([y1, y2])
            min_y = int(image.shape[0] * (1 / 2)) #the last 4/7th of the image
            max_y = int(image.shape[0])
    poly_left = np.poly1d(np.polyfit(
        left_line_y,
        left_line_x,
        deg=1
    ))

    left_x_start = int(poly_left(max_y))
    left_x_end = int(poly_left(min_y))

    poly_right = np.poly1d(np.polyfit(
        right_line_y,
        right_line_x,
    deg=1
    ))

    right_x_start = int(poly_right(max_y))
    right_x_end = int(poly_right(min_y))
    line_image = draw_lines(
        image,
        [[
            [left_x_start, max_y, left_x_end, min_y],
            [right_x_start, max_y, right_x_end, min_y],
        ]],
        thickness=30,
    )
    return line_image, left_x_start,left_x_end, right_x_start, right_x_end,min_y,max_y

# Function to generate a binary mask for the lane based on line endpoints and thickness
def generate_lane_mask(image, left_x_start, left_x_end, right_x_start, right_x_end, max_y, min_y, thickness=30):
    # Create a blank binary mask (same size as the image)
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # Draw the predicted lanes on the mask (left and right lanes)
    cv2.line(mask, (int(left_x_start), max_y), (int(left_x_end), min_y), (255), thickness)
    cv2.line(mask, (int(right_x_start), max_y), (int(right_x_end), min_y), (255), thickness)
    
    return mask

# Function to calculate the F1 score based on pixel overlap
def calculate_pixel_overlap(predicted_mask, ground_truth_mask):
    predicted_flat = predicted_mask.flatten()
    ground_truth_flat = ground_truth_mask.flatten()

    # Calculate TP,FP,FN
    TP = np.sum((predicted_flat == 255) & (ground_truth_flat == 255))  # Predicted and ground truth are both white (255)
    FP = np.sum((predicted_flat == 255) & (ground_truth_flat == 0))    # Predicted is white, ground truth is black (0)
    FN = np.sum((predicted_flat == 0) & (ground_truth_flat == 255))    # Predicted is black, ground truth is white

    # Calculate the F1 score
    if TP + FP + FN == 0:
        return 0.0  # Avoid division by zero if there is no overlap at all
    F1 = 2 * TP / (2 * TP + FP + FN)
    return F1


# Function to parse lane data from .lines.txt file (using the middle two lanes)
def parse_middle_lanes(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    if len(lines) < 3:
        raise ValueError("Annotation file doesn't contain enough lane data.")

    left_lane, right_lane = [], []

    # Extracting the second left lane (index 1) and the second right lane (index 2)
    left_line = list(map(float, lines[0].split()))  # This picks the second left lane
    right_line = list(map(float, lines[2].split()))  # This picks the second right lane

    left_lane = [(left_line[i], left_line[i+1]) for i in range(0, len(left_line), 2)]  # (x, y) pairs for the second left lane
    right_lane = [(right_line[i], right_line[i+1]) for i in range(0, len(right_line), 2)]  # (x, y) pairs for the second right lane

    return left_lane, right_lane

# Function to draw lane with line instead of circles and set thickness
def draw_lane(image, left_x_start, left_x_end, right_x_start, right_x_end, max_y, min_y, color_left, color_right, thickness=30):
    # Plot left and right lanes by drawing lines between start and end points
    cv2.line(image, (int(left_x_start), max_y), (int(left_x_end), min_y), color_left, thickness)
    cv2.line(image, (int(right_x_start), max_y), (int(right_x_end), min_y), color_right, thickness)




#Retinex Code
def Single_SR(I,sigma_g):
	x = np.linspace(-200,200,401)
	y = np.linspace(-200,200,401)
	x = np.reshape(x,[401,1])
	y = np.reshape(y,[1,401])
	G = np.exp(-((x**2)+(y**2))/(2*(sigma_g**2)))
	G = (1/np.sum(G))*G
	L = np.zeros_like(I)
	if(len(I.shape)==3):
		for i in range(3):
			L[:,:,i] = signal.fftconvolve(I[:,:,i],G,mode='same')
	else:
		L = signal.fftconvolve(I,G,mode='same')		
	SSR = np.log(I+1) - np.log(L+1)
	SSR = 255*(SSR-np.min(SSR))/(np.max(SSR)-np.min(SSR)+1)
	return SSR

def Gamma_Correction(I,gamma):
	corrected = np.power(I,gamma)
	return corrected

def Channel_Chromaticity(I):
	beta = 46
	alpha = 125
	red_channel = I[:,:,0]
	green_channel = I[:,:,1]
	blue_channel = I[:,:,2]
	total = red_channel+green_channel+blue_channel
	total = np.stack([total,total,total],axis = -1)
	J = beta * (np.log(alpha * I+1) - np.log(total+1))
	J = 255*(I-np.min(J))/(np.max(J)-np.min(J)+1)
	return J

def Normalization(I):
	G = 20
	b = -6
	J = 255*(I-np.min(I))/(np.max(I)-np.min(I)+1)
	# J = G * (I - b)
	J = np.clip(J, 0, 255)
	return J

def histogram_equalization(image):
    r, g, b = cv2.split(image)
    r_eq = cv2.equalizeHist(r)
    g_eq = cv2.equalizeHist(g)
    b_eq = cv2.equalizeHist(b)
    equalized_image = cv2.merge([r_eq, g_eq, b_eq])
    return equalized_image

def FSCS(img):
	img_float = np.float32(img)
	min_val = np.nanmin(img_float)
	max_val = np.nanmax(img_float)
	fscs_img = (img_float - min_val) / (max_val - min_val) * 255
	fscs_img = np.uint8(fscs_img)
	return fscs_img

def MSRCP(I):
	MSRCP_R = np.zeros_like(I[:,:,0])
	MSRCP_B = np.zeros_like(I[:,:,0])
	MSRCP_G = np.zeros_like(I[:,:,0])
	int_I = (I[:,:,0]+I[:,:,1]+I[:,:,2])/3
	int_MSR = (Single_SR(int_I,15)+Single_SR(int_I,80)+Single_SR(int_I,250))/3
	int_I_1 = color_balance(int_MSR,1,1)
	dim = I.shape
	for i in range(0,dim[0],1):
		for j in range(0,dim[1],1):
				B = np.max(I[i,j])
				A = np.nanmin([np.float32(255/B),int_I_1[i,j]/int_I[i,j]])
				MSRCP_R[i,j] = A*I[i,j,0]
				MSRCP_B[i,j] = A*I[i,j,1]
				MSRCP_G[i,j] = A*I[i,j,2]
	MSRCP = np.stack([MSRCP_R,MSRCP_B,MSRCP_G],axis=-1)
	MSRCP = 255*(MSRCP-np.nanmin(MSRCP))/(np.nanmax(MSRCP)-np.nanmin(MSRCP)+1)
	# print(MSRCP)
	MSRCP = np.clip(MSRCP,0,255)
	return MSRCP

def color_balance(img, low_per, high_per):
    tot_pix = img.shape[1] * img.shape[0]
    low_count = tot_pix * low_per / 100
    high_count = tot_pix * (100 - high_per) / 100
    ch_list = []
    if len(img.shape) == 2:
        ch_list = [img]
    else:
        ch_list = cv2.split(img)

    cs_img = []
    for i in range(len(ch_list)):
        ch = ch_list[i].astype('uint8')
        cum_hist_sum = np.cumsum(cv2.calcHist([ch], [0], None, [256], (0, 256)).flatten())
        li, hi = np.searchsorted(cum_hist_sum, (low_count, high_count))
        if li == hi:
            cs_img.append(ch)
            continue
        lut = np.array([0 if i < li else (255 if i > hi else round((i - li) / (hi - li) * 255))
                        for i in np.arange(0, 256)], dtype='uint8')
        cs_ch = cv2.LUT(ch, lut)
        cs_img.append(cs_ch)
    if len(cs_img) == 1:
        return np.squeeze(cs_img)
    elif len(cs_img) > 1:
        return cv2.merge(cs_img)
    return None