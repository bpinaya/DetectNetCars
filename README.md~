##Udacity P4 Line Detection

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./readme_imgs/distortion.png "Distortion Corrected"
[image12]: ./readme_imgs/distortion2.png "Distortion Corrected2"
[image2]: ./readme_imgs/roi.png "Region of Interest"
[image3]: ./readme_imgs/thresh.png "Binary Example"
[image4]: ./readme_imgs/pipeline.png "Pipeline on Image"
[video1]: ./project_video_output.mp4 "Output Video"

### [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
---
###Camera Calibration

####1. Since the camera calibration is run only once, a lot of optimization can be run on this routine, specially `cornerSubPix`, which improves the chessboard corners found. Using `cornerSubPix` requires a criteria, more info can be read [here](http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html).

After we have the criteria and both the structures needed to store the points Glob is used to load the images easily and run the `findChessboardCorners` routine. The camera calibration function is the following:

```python
def camera_calibration(num_corners = CORNERS_X_Y):
    # http://docs.opencv.org/2.4/modules/imgproc/doc/feature_detection.html?highlight=cornersubpix
    # http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.0001)
    objp = np.zeros((num_corners[0]*num_corners[1], 3), np.float32)
    objp[:,:2] = np.mgrid[0:num_corners[0],0:num_corners[1]].T.reshape(-1,2)

    object_points = []
    image_points = []
    image_size = None
    images = glob.glob(CAMERA_CAL_PATH)
    for idx, fname in enumerate(images):
        image = cv2.imread(fname)
        
        gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image_size = gray.shape

        ret, corners = cv2.findChessboardCorners(gray, num_corners, None)

        if ret:
            object_points.append(objp)
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            image_points.append(corners2)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(object_points, image_points, image_size[::-1], None, None)

    return mtx, dist
```

That returns `mtx` and the `dist`, both needed to have an undistorted image, for example this one:

![alt text][image1]

That has been greatly corrected and modified.

###Pipeline (single images)

####1. Correcting image distortion.

Using the parameters calculated above, the undistorted image of the road pictures shows:
![alt text][image12]

####2. Thresholding
Most of the thresholding functions were provided during the module, some minor changes added, the following functions are used for thresholding:
```python
def abs_sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    if orient=='x':
        img_s = cv2.Sobel(image,cv2.CV_64F, 1, 0)
    else:
        img_s = cv2.Sobel(image,cv2.CV_64F, 0, 1)
    img_abs = np.absolute(img_s)
    img_sobel = np.uint8(255*img_abs/np.max(img_abs))
    binary_output = np.zeros_like(img_sobel)
    binary_output[(img_sobel >= thresh[0]) & (img_sobel <= thresh[1])] = 1
    return binary_output

def mag_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    img_sx = cv2.Sobel(image,cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    img_sy = cv2.Sobel(image,cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    img_s = np.sqrt(img_sx**2 + img_sy**2)
    img_s = np.uint8(img_s*255/np.max(img_s))
    binary_output = np.zeros_like(img_s)
    binary_output[(img_s>=thresh[0]) & (img_s<=thresh[1]) ]= 1
    return binary_output

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    img_sx = cv2.Sobel(image,cv2.CV_64F,1,0, ksize=sobel_kernel)
    img_sy = cv2.Sobel(image,cv2.CV_64F,0,1, ksize=sobel_kernel)
    grad_s = np.arctan2(np.absolute(img_sy), np.absolute(img_sx))
    binary_output = np.zeros_like(grad_s)
    binary_output[(grad_s>=thresh[0]) & (grad_s<=thresh[1])] = 1
    return binary_output
```

And output of the combination of those three is shown in the next image
![alt text][image3]

####3. Region of Interest

To get our region of interest, we need to have a warped image from the birdeye perspective, that is achieved in the `get_roi_transformed` function, that goes as follow:

```python
def get_roi_transformed(image):
    image_size = image.shape
    roi_y_bottom = np.uint(image_size[0])
    roi_y_top = 450# np.uint(image_size[0]/1.5)
    roi_center_x = np.uint(image_size[1]/2)
    roi_x_top_left = roi_center_x - .25*roi_center_x
    roi_x_top_right = roi_center_x + .25*roi_center_x
    roi_x_bottom_left = 0
    roi_x_bottom_right = np.uint(image_size[1])
    # print(image_size,roi_y_bottom,roi_y_top,roi_center_x,roi_x_top_left,roi_x_top_right,roi_x_bottom_left,roi_x_bottom_right)
    src = np.float32([[roi_x_bottom_left,roi_y_bottom],[roi_x_bottom_right,roi_y_bottom],[roi_x_top_right,roi_y_top],[roi_x_top_left,roi_y_top]])
    dst = np.float32([[0,image_size[0]],[image_size[1],image_size[0]],[image_size[1],0],[0,0]])
    warped_image, M_warp, Minv_warp = warp_image(image,src,dst,(image_size[1],image_size[0]))
    return warped_image, M_warp, Minv_warp
```

Our generated Region of interest will be:


![alt text][image2]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:

![alt text][image5]

####5. Curvature
From the formula provided [here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) the function `get_curvature` is defined, to return the curvature given our polynomial:
```python
def get_curvature(polynomial,y):
    A = polynomial[0]
    B = polynomial[1]
    return (1+(2*A*y+B)**2)**1.5/2/A
```

####6. Image Pipeline
The function `detect_lane_pipeline` is the main function for the line detection, here we take into consideration if we are on the first frame, or else if information from the past last frame should be used in order to speed up things. The sobel filters are applied on the ROI, the lines are detected with histograms peaks, then data is reused sometimes if it's other frame than the first one.

Our output can be seen here:

![alt text][image4]

That clearly shows the lanes are recognized and the curvature is being obtained.
---

###Pipeline (video)

Here's a [link to my video result](./project_video_final.mp4)

---

###Discussion
This module was really clear, with my background on CV and also the lessons given, it was fairly easy to reach a solution, but this solution shows it's limitations, specially during sharp turns, or abrupt changes in the lanes, other conditions like more shadows, ice, fog, or a extremely bright day have not being taken into account, also this is only tested with images taken at day, not at night, and more improvements should be performed.

###Acknowledgments
To the Udacity Slack Channel, for all the help and ideas.
