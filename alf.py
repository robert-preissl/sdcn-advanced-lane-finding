
from PIL import Image

import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import tensorflow as tf
tf.python.control_flow_ops = tf # for https://github.com/fchollet/keras/issues/3857

# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML
import imageio

# some runtime parameters to be set via command line
flags = tf.app.flags
FLAGS = flags.FLAGS

flags.DEFINE_integer('do_calibration', '0', "Do a camera calibration. if not, read pickled calibration matrix.")
flags.DEFINE_integer('use_test_images', '0', "Use test images. if not, read the project video.")
flags.DEFINE_string('test_image_id', '1', "id of test image to use.")

# Usage:
#  - for test images   : python alf.py --do_calibration 0 --use_test_images 1 --test_image_id 1
#  - for project video : python alf.py --do_calibration 0 --use_test_images 0


# a class to store some of the characteristics of lane lines (left and right combined)
class LaneLines():
    def __init__(self, confidence, best_fit_left, best_fit_right, radius_of_curvature_left, radius_of_curvature_right, line_base_pos, allx_left, allx_right, ally_left, ally_right, left_start_index, right_start_index):
        self.confidence = confidence # 0.0 low confidence / 1.0 highest confidence that these are good lane lines
        # polynomial coefficients for the most recent fit
        self.best_fit_left = best_fit_left
        self.best_fit_right = best_fit_right
        # radius of curvature of the lines in Meters
        self.radius_of_curvature_left = radius_of_curvature_left
        self.radius_of_curvature_right = radius_of_curvature_right
        # distance in meters of vehicle center from the line
        self.line_base_pos = line_base_pos
        # x values for detected line pixels
        self.allx_left = allx_left
        self.allx_right = allx_right
        # y values for detected line pixels
        self.ally_left = ally_left
        self.ally_right = ally_right
        # x starting positions to start lane search from
        self.left_start_index = left_start_index
        self.right_start_index = right_start_index

    # confidence will be boosted by computing correct lanes (in terms of right distance and if parallel)
    #  and confidence is lost if we are not happy with our computed lanes (in terms of right distance and if parallel)
    def update_confidence(self, lane_check):
        if lane_check:
            if self.confidence < 0.95: # 1 .. max confidence
                self.confidence = self.confidence + 0.1
        else:
            if self.confidence > 0.05: # 0 .. min confidence
                self.confidence = self.confidence - 0.1

    # we are confident when the confidence level is above a certain threshold
    def check_confidence(self):
        if self.confidence > 0.95:
            return True
        return False


# Compute the camera calibration matrix and distortion coefficients given a set of chessboard images
# return: mtx, dist
def get_calibr_matrix_and_distortion_coeff(nx, ny):
    # read the calibration images and store them in a list:
    images = glob.glob("./CarND-Advanced-Lane-Lines-master/camera_cal/calibration*.jpg")
    print("read calibration images. images.len = ",len(images))

    # arrays to store objects points and image points for all calibration images
    objpoints = [] # 3d points in real space
    imgpoints = [] # 2d points in image plane

    objp       = np.zeros((nx*ny,3), np.float32)
    objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2) # x,y coordinates

    for fname in images:
        # read in each calibration image
        img = mpimg.imread(fname)

        # convert image to Grayscale
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
        print("Find chessboard corner for fname = ", fname, " ret = ", ret)

        # Note, a few of the images have ret=False when nx=9, ny=6 (which yields best results)
        #  but most have ret=True, so ignore this so far.

        # if corners are found, add object points, image points
        if ret == True:
            imgpoints.append(corners)
            objpoints.append(objp)

            # draw and display the corners
            img = cv2.drawChessboardCorners(img, (nx,ny), corners, ret)

    # calibrate the camera:
    # a) read a test image (take one of them. should be ok since the same distortion should appy to each image)
    test_img = mpimg.imread('./CarND-Advanced-Lane-Lines-master/test_images/test1.jpg')
    # b) convert to grayscale
    test_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, test_gray.shape[::-1],None,None)

    # save as pickle file
    mtx_dist = { "mtx": mtx, "dist": dist }
    pickle.dump( mtx_dist, open( "wide_dist_pickle.p", "wb" ) )

    return mtx, dist


# Apply the distortion correction to a raw image
# return: the Undistorted image
def apply_distortion_correction(image, mtx, dist,plot=False):
    undist = cv2.undistort(image, mtx, dist, None, mtx)

    if plot:
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
        f.tight_layout()
        ax1.imshow(image)
        ax1.set_title('Original Image', fontsize=50)
        ax2.imshow(undist)
        ax2.set_title('Undistorted Image', fontsize=50)
        plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
        plt.show()

    return undist

# Warp an image based on hardcoded (for 960x540 and for 1280x720 images) source and destination points
# return: a warped image
def warp_image(img, low_res=False):
    offset = 100 # offset for dst points
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Grab the image shape
    img_size = (gray.shape[1], gray.shape[0])

    src = None
    if low_res:
        # for the 960x540 images:
        src = np.float32([(15,img_size[1]),(381, 355), (584, 355), (960,img_size[1])])
    else:
        # for the 1280x720 images:
        src = np.float32([(230,670),(545, 478), (775, 478), (1150, 670)])

    print( "warp_image: src = ", src )

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    offset = 0 # offset for dst points
    dst = np.float32([[0 + offset, img_size[1] - offset],
                      [0 + offset, 0 + offset],
                      [img_size[0] - offset, 0 + offset],
                      [img_size[0] - offset, img_size[1] - offset]])
    print( "warp_image: dst = ", dst )

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)
    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(img, M, img_size)

    return warped, M

# Apply color mask to extract yellow and white features of an image
# return: a resulting image with yellow and white features extracted
def apply_color_masks(img, white_mask=1):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # define range of yellow color in HSV
    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Threshold the HSV image to get only yellow colors
    yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)
    res_yellow  = cv2.bitwise_and(img,img, mask = yellow_mask)

    lower_white = None
    upper_white = None
    if white_mask == 1:
        lower_white  = np.array([  20,   0, 200])
        upper_white  = np.array([ 255,  80, 255])
    else:
        lower_white  = np.array([  0,   0, 200])
        upper_white  = np.array([ 180,  255, 255])

    white_mask = cv2.inRange(hsv, lower_white, upper_white)
    res_white  = cv2.bitwise_and(img,img, mask = white_mask)

    res = cv2.add(res_yellow,res_white)

    return res

# Apply x or y gradients using the sobel functions
# return: a binary image with x, y gradients above a certain threshold
def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Apply x or y gradient with the OpenCV Sobel() function
    # and take the absolute value
    if orient == 'x':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel))
    if orient == 'y':
        abs_sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel))
    # Rescale back to 8 bit integer
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # Create a copy and apply the threshold
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    return binary_output

# Apply x or y gradient magnitudes
# return: a binary image with x, y gradient magnitudes within a certain threshold band
def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output

# Apply a "direction of the gradient" filter on an image
# return: a binary image with direction of gradient fulfilling a certain radial threshold
def dir_threshold(img, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # Calculate the x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the gradient direction,
    # apply a threshold, and create a binary image result
    absgraddir = np.arctan2(np.absolute(sobely), np.absolute(sobelx))
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    return binary_output

# Apply the x & y gradients PLUS the magnitude and directional gradients
# return: binary output meeting threshold criteria
def apply_combined_gradients(image, x_tresh=(20,100), y_tresh=(20,100), mag_thresh_=(30, 100), dir_tresh=(0.7, 1.3)):
    # Choose a Sobel kernel size
    ksize = 15

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=ksize, thresh=x_tresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=ksize, thresh=y_tresh)
    mag_binary = mag_thresh(image, sobel_kernel=ksize, mag_thresh=mag_thresh_)
    dir_binary = dir_threshold(image, sobel_kernel=ksize, thresh=dir_tresh)
    # Try different combinations and see what you get. For example, here is a selection for pixels
    # where both the x and y gradients meet the threshold criteria, or the gradient magnitude and
    #  direction are both within their threshold values.

    combined = np.zeros_like(dir_binary)
    # combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1))] = 1
    combined[((mag_binary == 1) | (dir_binary == 1))] = 1

    return combined

# Apply coloring and gradient filtering on an image
# return: binary output meeting threshold criteria
def combined_gradient_and_color(img, sobel_kernel=15, s_thresh=(90, 255), l_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hsv[:,:,0]
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    lower_yellow = np.array([20, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # Sobel x
    # We will apply Sobel filters and threshold the magnitude of the gradients in x- and y- directions
    #  for S and L channels of HLS image.
    sobelx = cv2.Sobel(s_channel, cv2.CV_64F, 1, 0, ksize=sobel_kernel) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

    # Threshold color channel
    l_binary = np.zeros_like(s_channel)
    s_binary = np.zeros_like(s_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Combine the binary thresholds
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(l_binary == 1) | (s_binary == 1) | (sxbinary == 1)  ] = 1

    return combined_binary


# Compute starting x- positions (for left & right lanes) to start the sliding window lane line search
#  using a histogram approach
# return: 2 starting positions for x. for left and right lanes
def histo_starting_indices(img, plotting):
    # rows are the first dimension
    histogram = np.sum(img[img.shape[0]/2:,:], axis=0)
    histogram = histogram / float(img.shape[0]/2)

    if plotting:
        plt.plot(histogram)
        plt.show()

    # the two most prominent peaks in this histogram will be good indicators of the x-position
    # of the base of the lane lines. I can use that as a starting point for where to search for the lines

    # find starting pos:
    length = len(histogram)

    left_index = 0
    right_index = 0
    index = 0
    max = 0
    for index in range(0, int(length/2)):
        if histogram[index] > max:
             max = histogram[index]
             left_index = index

    index = 0
    max = 0
    for index in range(int(length/2), length):
        if histogram[index] > max:
             max = histogram[index]
             right_index = index

    print("left_index = ", left_index, " / right_index = ", right_index)

    return left_index, right_index

# Compute the number of pixels in a certain row, within certain columns (based on a window size)
# return: integer value of pixel intensity
def get_pixel_intensity(bin_img, row_index, col_index, window_size):

    # out of bounds checks:
    if (col_index < 0 or ((col_index+window_size) >= bin_img.shape[1])) :
        # print(" \n \n OUT OF BOUNDS. row_index = ", row_index, " / col_index = ", col_index, " / window_size = ", window_size)
        return -1

    window = bin_img[row_index,col_index:col_index+window_size]
    pixel_intensity = np.sum(window)

    return pixel_intensity

# determine the starting pixel of a row. which determines the window with the most
#  pixels equal 1 (best window is parametrized: determined by a starting_index + window size)
# return: the ideal starting index for a certain row
def get_starting_index_row(bin_img, col_index_start, col_index_end, row_index, window_size, debug = 0):

    if debug == 1:
        print("  1  col_index_start: ", col_index_start, " / col_index_end = ", col_index_end)

    # get the row with index "row_index"
    row = bin_img[row_index,:]

    best_col_index = col_index_start
    best_col_indeces = []
    max_pixel_intensity = -1
    only_zeros_in_row = True
    for col_index in range(col_index_start, col_index_end, 1): # range(s_index_col, s_index_col + window_size, 1):
        # get the pixel intensity for each window
        pixel_intensity = get_pixel_intensity(bin_img, row_index, col_index, window_size)
        if pixel_intensity > max_pixel_intensity:
            max_pixel_intensity = pixel_intensity
            best_col_index = col_index
            best_col_indeces = []
            if pixel_intensity > 0:
                only_zeros_in_row = False
        if pixel_intensity == max_pixel_intensity:
            best_col_indeces.append(col_index)

    # if no 1 in a row, do not average, return the given starting index
    if only_zeros_in_row:
        if debug == 1:
            print("  2 only zeros:  best_col_index: ", col_index_start)
        return col_index_start + window_size

    # when equal intensities:
    best_avg_index = int( sum(best_col_indeces) / len(best_col_indeces) )
    if debug == 1:
        print("  2  best_col_index: ", best_col_index, " / best_col_indeces = ", best_col_indeces, " avg best index = ", best_avg_index )
    return best_avg_index

# Construct a lane image by assempling the pixels found by the sliding window lane finding approach
# return: a binary image with lane pixels
def get_lane_image(bin_img, starting_index, left = 1, window_size = 40, set_start_pixel = True):
    lane_img = np.zeros_like(bin_img)
    shape = bin_img.shape
    rows = shape[0]
    cols = shape[1]

    if set_start_pixel:
        lane_img[rows-1][starting_index] = 1

    # for each row, do a sliding window to find the window with high pixel intensity
    start_index = starting_index - window_size + 1 # start_index is the leftmost pixel to start the sliding window

    for row_index in range(rows-1, 0, -1):
        end_index = min(cols-1, start_index + 2 * window_size)
        start_index = get_starting_index_row(bin_img, start_index, end_index, row_index, window_size)

        # copy pixel elements into the lane image (only if pixel intensity > 0; which means)
        lane_img[row_index,start_index:start_index+window_size] = bin_img[row_index,start_index:start_index+window_size]
        start_index = start_index - window_size

    return lane_img


# Computes the polynomial fit for a lane (black/white) pixel image
# return: x and y indices of 1-pixels and the polynomial fit
def get_lane_polynomial(img):
    rows, cols = img.shape
    indices = np.vstack(np.unravel_index(np.arange(rows*cols), (rows, cols))).T
    # extract the pixel value (1/0) incl. the index
    value_and_index = np.hstack((img.reshape(rows*cols,1), indices))

    # img = array([[1, 2],
    #              [3, 4],
    #              [5, 6]])

    # first column is pixel value : 0/1 , then index 0,0 etc.
    #
    # array([[1, 0, 0],
    #    [2, 0, 1],
    #    [3, 1, 0],
    #    [4, 1, 1],
    #    [5, 2, 0],
    #    [6, 2, 1]])

    # only keep the indeces which have a 1 pixel
    value_1_and_index = value_and_index[value_and_index[:,0] == 1]

    y_indeces = value_1_and_index[:,1]
    x_indeces = value_1_and_index[:,2]

    fit = np.polyfit(y_indeces, x_indeces, 2)

    # make sure we extrapolate the full y-range
    y_indeces[0] = 1
    y_indeces[len(y_indeces)-1] = rows - 1
    fitx = fit[0]*y_indeces**2 + fit[1]*y_indeces + fit[2]

    return x_indeces, y_indeces, fit, fitx

# Plot the polynomial lane fits
# return: plot
def plot_polynomial_lane_fits(img, left_x_indeces, left_y_indeces, left_fitx, right_x_indeces, right_y_indeces, right_fitx):
    rows, cols = img.shape
    plt.plot(left_x_indeces, left_y_indeces, 'o', color='red')
    plt.plot(left_fitx, left_y_indeces, color='green', linewidth=3)

    plt.plot(right_x_indeces, right_y_indeces, 'o', color='blue')
    plt.plot(right_fitx, right_y_indeces, color='green', linewidth=3)

    plt.xlim(0, cols)
    plt.ylim(0, rows)
    plt.gca().invert_yaxis()
    plt.show()


# with the polynomial fits we can calculate the radius of curvatures
# return: left- and right-curvature radius
def get_curvature_radius(left_yvals, right_yvals, left_fit, right_fit):
    # Define y-value where we want radius of curvature
    # I'll choose the maximum y-value, corresponding to the bottom of the image
    y_eval_left = np.max(left_yvals)
    y_eval_right = np.max(right_yvals)
    left_curverad = ((1 + (2*left_fit[0]*y_eval_left + left_fit[1])**2)**1.5) \
                                 /np.absolute(2*left_fit[0])
    right_curverad = ((1 + (2*right_fit[0]*y_eval_right + right_fit[1])**2)**1.5) \
                                    /np.absolute(2*right_fit[0])
    print("left_curverad = ", left_curverad, " / right_curverad = ", right_curverad)
    return left_curverad, right_curverad


# Calculation of radius of curvature after correcting for scale in x and y
# return: left- and right-curvature radius in Meters
def get_curvature_radius_meters(leftx, rightx, left_yvals, right_yvals):
    # Define conversions in x and y from pixels space to meters
    ym_per_pix = 30/720.0 # meters per pixel in y dimension
    xm_per_pix = 3.7/700.0 # meteres per pixel in x dimension

    y_eval_left = np.max(left_yvals)
    y_eval_right = np.max(right_yvals)

    left_fit_cr = np.polyfit(left_yvals*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(right_yvals*ym_per_pix, rightx*xm_per_pix, 2)

    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval_left + left_fit_cr[1])**2)**1.5) \
                                 /np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval_right + right_fit_cr[1])**2)**1.5) \
                                    /np.absolute(2*right_fit_cr[0])
    # Now our radius of curvature is in meters
    print("left_curverad [m] = ", left_curverad, " / right_curverad [m] = ", right_curverad)
    return left_curverad, right_curverad

# Draw the lane lines back on the road
# return: an image with lane lines drawn on it
def draw_lines_on_road(perspective_M, orig_image, warped, left_fitx, right_fitx, left_y_indeces, right_y_indeces):
    # Create an image to draw the lines on
    warp_zero = np.zeros_like(warped).astype(np.uint8)
    color_warp = warp_zero# np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, left_y_indeces]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, right_y_indeces])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    Minv = cv2.invert(perspective_M)
    newwarp = cv2.warpPerspective(color_warp, Minv[1], (orig_image.shape[1], orig_image.shape[0]))#, flags= cv2.WARP_INVERSE_MAP)
    # Combine the result with the original image
    result = cv2.addWeighted(orig_image, 1, newwarp, 0.3, 0)
    return result

# Compute the position of the car in regard to the center position
# return: car distance in meters from the center
def get_car_position(img, left_lane_pixel, right_lane_pixel):
    # meters from center
    xm_per_pix = 3.7/700 # meters per pixel in x dimension
    screen_middel_pixel = img.shape[1]/2
    car_middle_pixel = int((right_lane_pixel + left_lane_pixel)/2)
    screen_off_center = screen_middel_pixel-car_middle_pixel
    meters_off_center = xm_per_pix * screen_off_center

    return meters_off_center

# Helper function to plot an BGR image
def plot_cv(img):
    b,g,r = cv2.split(img)
    img2 = cv2.merge([r,g,b])
    plt.imshow(img2) # expect true color
    plt.show()

# Helper function to plot an BGR image and a grayscale image
def combo_plot(imgA, captionA, imgB, captionB):
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    b,g,r = cv2.split(imgA)
    imgA2 = cv2.merge([r,g,b])
    ax1.imshow(imgA2)
    ax1.set_title(captionA, fontsize=50)

    ax2.imshow(imgB, cmap='gray')
    ax2.set_title(captionB, fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

# Helper function to plot 3 grayscale images
def triple_plot_gray(imgA, captionA, imgB, captionB, imgC, captionC):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(imgA, cmap='gray')
    ax1.set_title(captionA, fontsize=30)
    ax2.imshow(imgB, cmap='gray')
    ax2.set_title(captionB, fontsize=30)
    ax3.imshow(imgC, cmap='gray')
    ax3.set_title(captionC, fontsize=30)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

# Helper function to plot the result image plus some images for diagnostics plus some distance information
def multi_plot_small(result, orig_image, combined_gradient_and_color_img, left_curverad_m, right_curverad_m, car_position, plotting): # left_lane_img, right_lane_img, poly_lines_img):#, diag4, diag5, diag6): #, diag7, diag8, diag9):
    # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 2880, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curvature[m]: left = ' + str(left_curverad_m) + ' / right = ' + str(right_curverad_m), (30, 60), font, 1, (255,0,0), 2)
    cv2.putText(middlepanel, 'Estimated car distance right of center[m] = ' + str(car_position), (30, 90), font, 1, (255,0,0), 2)

    result_image = np.zeros((660, 2880, 3), dtype=np.uint8)
    result_image[0:540, 0:960] = result
    result_image[0:540, 960:1920] = orig_image
    result_image[0:540, 1920:2880, 2] = combined_gradient_and_color_img
    result_image[540:660, 0:2880] = middlepanel

    if plotting:
        plot_cv(result_image)

    return result_image

# Helper function to plot the result image plus some images for diagnostics plus some distance information
def multi_plot(result, orig_image, combined_gradient_and_color_img, left_curverad_m, right_curverad_m, car_position, plotting): # left_lane_img, right_lane_img, poly_lines_img):#, diag4, diag5, diag6): #, diag7, diag8, diag9):
    # middle panel text example
    # using cv2 for drawing text in diagnostic pipeline.
    font = cv2.FONT_HERSHEY_COMPLEX
    middlepanel = np.zeros((120, 3840, 3), dtype=np.uint8)
    cv2.putText(middlepanel, 'Estimated lane curvature[m]: left = ' + str(left_curverad_m) + ' / right = ' + str(right_curverad_m), (30, 60), font, 1, (255,0,0), 2)
    cv2.putText(middlepanel, 'Estimated car distance right of center[m] = ' + str(car_position), (30, 90), font, 1, (255,0,0), 2)

    result_image = np.zeros((840, 3840, 3), dtype=np.uint8)
    result_image[0:720, 0:1280] = result
    result_image[0:720, 1280:2560] = orig_image
    result_image[0:720, 2560:3840, 2] = combined_gradient_and_color_img
    result_image[720:840, 0:3840] = middlepanel

    if plotting:
        plot_cv(result_image)

    return result_image



# --- the advanced-lane-finding pipeline ---
# return: a multi-plot (result image) + diagnostics, a result image, and a LaneLines object
def alf_pipeline(image, mtx, dist, prev_lane, plotting, low_res):
        # plot_cv(image)

        # 1. Apply the distortion correction to the raw image
        undist_img = apply_distortion_correction(image, mtx, dist, plot=False)
        if plotting:
            plot_cv(undist_img)


        # 2. apply_perspective_transform to get a "birds-eye view"
        unwarped, perspective_M = warp_image(undist_img, low_res)
        if plotting:
            plot_cv(unwarped)


        # 3. Use color transforms, gradients, etc., to create a thresholded binary image.
        thresholded = apply_color_masks(unwarped)
        left_lane_pixels = np.sum(thresholded[:,:thresholded.shape[1]/2])
        right_lane_pixels = np.sum(thresholded[:,thresholded.shape[1]/2:])

        # for certain images, especially on gray concrete roads, right lane lane marks (in white) are not detected.
        # check if not detected and if yes, choose a different mask for white
        if right_lane_pixels < 1000:
            thresholded = apply_color_masks(unwarped, white_mask=2)

        if plotting:
            plot_cv(thresholded)

        combined_thresh = apply_combined_gradients(thresholded, x_tresh=(20,100), y_tresh=(20,100), mag_thresh_=(30, 100), dir_tresh=(0.7, 1.7))
        # Plot the result
        if plotting:
            combo_plot(thresholded, 'Orig. Image', combined_thresh, 'M/Dir grad. on col. thresh')


        # here, feed in the color image and return a color combined with gradient
        combined_gradient_and_color_img = combined_gradient_and_color(unwarped)
        # Plot the result
        if plotting:
            combo_plot(thresholded, 'Original Image', combined_gradient_and_color_img, 'Color and sobel Grad.')


        # Note: we can move on with "combined_thresh" or with "combined_gradient_and_color_img"
        color_and_gradient_image = combined_thresh


        # 4. detect lane pixels and fit to find lane boundary.

        # if we have achieved a high-confidence detection (i.e., confidence is above 0.8), that positional
        #  knowledge will be used as a starting point to find the lines. I.e., we save the time to do an
        #  expensive histogram search
        left_start_index = None
        right_start_index = None
        if prev_lane is not None and prev_lane.check_confidence():
            left_start_index = prev_lane.left_start_index
            right_start_index = prev_lane.right_start_index
        else:
            left_start_index, right_start_index = histo_starting_indices(color_and_gradient_image, plotting)
        print("SS1 -- left_start_index = ", left_start_index, " / right_start_index = ", right_start_index)

        left_lane_img = get_lane_image(color_and_gradient_image, left_start_index, left = 1, window_size = 100)
        right_lane_img = get_lane_image(color_and_gradient_image, right_start_index, left = 1, window_size = 100)
        # Plot the result
        if plotting:
            triple_plot_gray(color_and_gradient_image, 'color & gradient image', left_lane_img, 'left lane', right_lane_img, 'right lane')


        left_x_indeces, left_y_indeces, left_fit, left_fitx = get_lane_polynomial(left_lane_img)
        right_x_indeces, right_y_indeces, right_fit, right_fitx = get_lane_polynomial(right_lane_img)
        # Plot the result
        if plotting:
            plot_polynomial_lane_fits(left_lane_img, left_x_indeces, left_y_indeces, left_fitx, right_x_indeces, right_y_indeces, right_fitx)


        # 5. Determine curvature of the lane and vehicle position with respect to center
        left_curverad, right_curverad = get_curvature_radius(left_y_indeces, right_y_indeces, left_fit, right_fit)
        left_curverad_m, right_curverad_m = get_curvature_radius_meters(left_x_indeces, right_x_indeces, left_y_indeces, right_y_indeces)
        car_position = get_car_position(image, left_x_indeces[0], right_x_indeces[0])


        # 6. Warp the detected lane boundaries back onto the original image.
        mapped_lanes = draw_lines_on_road(perspective_M, undist_img, unwarped, left_fitx, right_fitx, left_y_indeces, right_y_indeces)


        # 7. Output visual display of the lane boundaries and numerical estimation of lane
        #  curvature and vehicle position.
        result_multi_plot = None
        if low_res:
            result_multi_plot = multi_plot_small(mapped_lanes, unwarped, color_and_gradient_image*255.0, left_curverad_m, right_curverad_m, car_position, plotting) # left_lane_img[1]*255.0, right_lane_img[1]*255.0, right_lane_img[1]*255.0)
        else:
            result_multi_plot = multi_plot(mapped_lanes, unwarped, color_and_gradient_image*255.0, left_curverad_m, right_curverad_m, car_position, plotting) # left_lane_img[1]*255.0, right_lane_img[1]*255.0, right_lane_img[1]*255.0)


        # Create LaneLines object and do confidence checks
        left_lane_starting_x_pos_polynom = left_fit[0]*670**2 + left_fit[1]*670 + left_fit[2]
        right_lane_starting_x_pos_polynom = right_fit[0]*670**2 + right_fit[1]*670 + right_fit[2]
        distance_between_polynomial_lane_lines_pixel = right_lane_starting_x_pos_polynom - left_lane_starting_x_pos_polynom
        distance_between_polynomial_lane_lines_meters = distance_between_polynomial_lane_lines_pixel*3.7/870.0

        left_lane_starting_x_pos_polynom_top = left_fit[2]
        right_lane_starting_x_pos_polynom_top = right_fit[2]
        distance_between_polynomial_lane_lines_pixel_top = right_lane_starting_x_pos_polynom_top - left_lane_starting_x_pos_polynom_top
        distance_between_polynomial_lane_lines_meters_top = distance_between_polynomial_lane_lines_pixel_top*3.7/700.0

        # check that a) lanes are separated by approximately the right distance horizontally and
        #            b) lanes are roughly parallel
        lane_parallel_check = abs(distance_between_polynomial_lane_lines_pixel - distance_between_polynomial_lane_lines_pixel_top) # diff the top and bottom horizontal lane distance
        lane_distance = distance_between_polynomial_lane_lines_pixel
        print("\n DD1 -- lane_distance = ", lane_distance, " / lane_parallel_check = ", lane_parallel_check)
        lane_check = False
        laneLines = None
        if prev_lane is not None:
            laneLines = LaneLines(prev_lane.confidence, left_fit, right_fit, left_curverad_m, right_curverad_m, car_position, left_fitx, right_fitx, left_y_indeces, right_y_indeces, left_start_index, right_start_index)
        else:
            laneLines = LaneLines(0.0, left_fit, right_fit, left_curverad_m, right_curverad_m, car_position, left_fitx, right_fitx, left_y_indeces, right_y_indeces, left_start_index, right_start_index)

        if lane_parallel_check < 100 and lane_distance > 1000 and lane_distance < 1100:
            lane_check = True
        laneLines.update_confidence(lane_check)

        print("\n EE1 -- laneLines.confidence = ", laneLines.confidence, " / laneLines.check_confidence = ", laneLines.check_confidence())

        return result_multi_plot, mapped_lanes, laneLines


def main():
    print("Start")

    mtx, dist = None, None
    # do the camera calibration only once and save the matrix and coefficients for later use.
    if FLAGS.do_calibration == 1:
        # compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
        mtx, dist = get_calibr_matrix_and_distortion_coeff(9,6)
    else:
        # read in the saved camera matrix and distortion coefficients
        dist_pickle = pickle.load( open( "wide_dist_pickle.p", "rb" ) )
        mtx = dist_pickle["mtx"]
        dist = dist_pickle["dist"]

    # we can run the advanced-lane-finding pipeline on a test-image or on a video stream
    if FLAGS.use_test_images == 1:
        test_img = cv2.imread('./CarND-Advanced-Lane-Lines-master/test_images/test' + FLAGS.test_image_id + '.jpg')
        new_image, mapped_lanes, laneLines = alf_pipeline(test_img, mtx, dist, None, True, False)
        plot_cv(mapped_lanes)
    else:
        reader = imageio.get_reader('./project_video.mp4')
        fps = reader.get_meta_data()['fps']
        writer_d = imageio.get_writer('./project_video_result_with_diagnostics.mp4', fps=fps)
        writer = imageio.get_writer('./project_video_result.mp4', fps=fps)
        cnt = 0
        confidence = 0.0
        laneLines = None
        for im in reader:
            if cnt % 1 == 0: #for testing we can just check 10% or smaller of the frames of the video.
                rgb_image = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                new_image, mapped_lanes, laneLines = alf_pipeline(rgb_image, mtx, dist, laneLines, False, False)
                print (" --> new_image = ", new_image.shape)
                b,g,r = cv2.split(new_image)
                new_image2 = cv2.merge([r,g,b])
                writer_d.append_data(new_image2[:, :, :])
                b,g,r = cv2.split(mapped_lanes)
                mapped_lanes2 = cv2.merge([r,g,b])
                writer.append_data(mapped_lanes2[:, :, :])
            cnt = cnt + 1
        writer_d.close()
        writer.close()
        print("Done: processed ", cnt, " many frames.")


if __name__ == "__main__":
    main()
