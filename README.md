# The ** Advanced Lane Finding project **

This is a discussion for the "advanced lane finding" exercise in Udacity's Self Driving Car Program.

[//]: # (Image References)
[undist]: ./undist.png
[bird_eye]: ./bird_eye.png
[color_mask_applied]: ./color_mask_applied.png
[mag_dir_gradients]: ./mag_dir_gradients.png
[histogram]: ./histogram.png
[left_and_right_lanes_binary_image]: ./left_and_right_lanes_binary_image.png
[polynomial_fit]: ./polynomial_fit.png
[diagnostic_view]: ./diagnostic_view.png
[projected_lane_lines]: ./projected_lane_lines.png
[histogram]: ./histogram.png

[video1]: ./project_video_result.mp4
[video2]: ./project_video_result_with_diagnostics.mp4


## *The Pipeline:*

The program to find lane lines can be run in two different ways:
* using test-images (useful for debugging)
* or on a video stream

The test-image, or each frame of the video will run through the "advanced lane finding pipeline".

These are the basic steps in the pipeline (which refer to comments in the "alf_pipeline" function of alf.py)

(Also, I did a plot for each of the steps when running a test-image through the pipeline. file-names are denoted in each step)

*  (1) Apply the distortion correction to the raw image:
  * In the main function we do the camera calibration only once (given a set of chessboard images) and save the matrix and coefficients for later use in a pickle file.
  * Via runtime parameters we can control if we want to do a camera calibration or if we want to read the pickled data.

  * Image-file for this step: "undist.png"

  ![alt text][undist]


*  (2) Apply_perspective_transform to get a "birds-eye view"
  * Warp an image based on hardcoded (for 960x540 and for 1280x720 images) source and destination points

  * Image-file for this step: "bird_eye.png"

  ![alt text][bird_eye]


*  (3) Use color transforms and gradients to create a thresholded binary image.
  * First, we apply yellow and white filters on the HSV converted image. (to extract yellow and white lane pixel features)
  * Second, we apply magnitude and directional gradients using a Sobel kernel of size 15.
  * The result is a new binary image

  * Image-files for this step: "color_mask_applied.png", "mag_dir_gradients.png"

  ![alt text][color_mask_applied]

  ![alt text][mag_dir_gradients]


*  (4) Detect lane pixels and fit to find lane boundary.
  * First, we use a "histogram based" approach to find the starting x-indeces for left/right lanes. (the x value corresponding to the highest peak is likely a good starting index for the lane line search)
  * Second, we do a "sliding window approach" starting at the above found starting index and find row by row the window with the most 1-pixels.
  * Third, we run a "polynomial fit" through the detected left and right lane binary pictures found in step 2. (a processing step has to happen before that, to convert the binary images of a lane into x and y coords)

  * Note: if we run the pipeline on a stream of video frames, if we have enough confidence to have strong lane detection we can reuse starting points from a previous lane and do not need to do the search for starting indeces. (More about the confidence of lane lines later)

  * Image-files for this step: "histogram.png", "left_and_right_lanes_binary_image.png", "polynomial_fit.png"

  ![alt text][histogram]

  ![alt text][left_and_right_lanes_binary_image]

  ![alt text][polynomial_fit]


*  (5) Determine curvature of the lane and vehicle position with respect to center
  * We compute the curvature in pixel and in Meters and also the car position in regard to the center.

  * Image-file for this step: "diagnostic_view.png"

  ![alt text][diagnostic_view]


*  (6) Warp the detected lane boundaries back onto the original image.

  * Image-file for this step: "projected_lane_lines.png"

  ![alt text][projected_lane_lines]


*  (7) Output visual display of the lane boundaries and numerical estimation of lane

  * Show the numerical values computed in step 5.) along with the picture of step 6.) and also show the "birds eye view" and the detected lane pixels. Such a diagnostic view is useful when debugging a video stream.

  * Image-file for this step: "diagnostic_view.png"

  ![alt text][diagnostic_view]


  * In this last step we also create a "LaneLines" object, which holds several properties of a lane line. One interesting feature of this class is that it keeps track of how well previous lanes were detected. In detail, we check at the every iteration of the pipeline:
    * if lanes are separated by approximately the right distance horizontally and
    * if lanes are roughly parallel

  If this is the case we can build up confidence (a simple float value between 0 and 1. 0 .. lowest confidence, 1 .. highest confidence). If this is not the case we loose confidence.
  If the confidence is built up very strongly we can use information of the laneLine object in a consequent pipeline iteration and for instance omit the step of finding starting x indeces for the lane line sliding window search. (we simply use the starting indeces of the previous iteration since the lane lines were very nicely detected over many iterations)


## *Comments about improvements, success factors, etc.*
  * One success factor definitely is to either output each step of the pipeline per video frame, or better, having one picture which includes many mosaic pieces of the pipeline. If a frame does not successfully show lane lines we can quickly see where it breaks
  * If something breaks, I created a test image in the right pixel size of a failing frame and just debug the image rather than a stream.

  * Improvements can definitely be made to support more tricky road conditions like rain, snow, or situations were no lanes are visible. Also situations where there is no clear view of the road could be tricky - like a big truck blocks the view.


**Result video**

  Here's a [link to my video result](./project_video_result.mp4)
  And here's a [link to my video result with diagnostics](./project_video_result_with_diagnostics.mp4)


 :tada:
