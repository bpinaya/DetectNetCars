## Udacity P5 Car Detection Using DIGITS and Caffe

---

**DriveNet with DIGITS and Caffe, to solve the last assigment of Udacity's Self Driving Car Nanodegree

The goals / steps of this project according the rubric are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

BUT (and here comes the interesting part), there are many ways to reach a goal, people from the Slack Channel have discussed about using a neural network to tackle this challenge, because the pipeline with HOG, a classifier with region proposals takes too long, specially for a real time solution. What will be proposed for this challenge, takes about 0.03 seconds to detect the cars on an image. Then again faster detection equals more detections per second, and with some tunning this can be easily deployed in a Jetson TX1.

So the goals I set myself for this challenge are the following:
* Use [DriveNet](https://arxiv.org/pdf/1312.6082.pdf) as an architecture to tackle the car detection problem.
* Use [Digits 5](https://developer.nvidia.com/digits) as a framework to train, this to facilitate production and to prove the potential of Digits for training of DNN, also the visualizations provided are top notch.
* Instead of the dataset provided from Udacity, use the [KITTI Dataset](http://www.cvlibs.net/datasets/kitti/) because of its size and potential scalability with Lidar Data.
* Use [Caffe](http://caffe.berkeleyvision.org/). Since we've already used Keras and TF, it'll be nice to test another framework, also for learning purposes.

[//]: # (Image References)

[detectnet1loss]: ./readme_imgs/detect1loss.png "Detectnet loss"
[detectnet1lr]: ./readme_imgs/detect1lr.png "Detectnet Learning rate"
[detectnet2loss]: ./readme_imgs/detect2loss.png "Region of Interest"
[detectnet2lr]: ./readme_imgs/detect2lr.png "Binary Example"
[detectnet]: ./readme_imgs/detectnet.png "DetectNet Architecture"
[coverage]: ./readme_imgs/coverage.png "coverage"
[inference]: ./readme_imgs/inference.png "inference"
[source]: ./readme_imgs/source.png "source"
[visual]: ./readme_imgs/visual.png "visual"
[digits]: ./readme_imgs/Capture.PNG "Digits"
[video1]: ./project_video_output.mp4 "Output Video"

---
### Framework Selection - Using DIGITS with Caffe

#### 1. DIGITS.

Digits is a framework by Nvidia, to facilitate training of deep neural networks. Currently it works only with Caffe and Torch. Its main advantage is the facility of training multiple models, loading new models and visualize results without a lot of effort, so that you can test a network to see if it would work in 10 minutes, and then improve on top of that. A glimpse into Digits can be seen on the next figure:

![alt text][digits]

Its interface is simple but powerful, it allows dataset selection, model selection and also a pre-trained model store, to use past models in other assignments. It usses Caffe, and the results can be expanded with [TensorRT](https://developer.nvidia.com/tensorrt) to deploy your models easily.

#### 2. KITTI Database. 

The database used from the assignment is taken from [KITTI](http://www.cvlibs.net/datasets/kitti/), having 12GB of data seemed useful for our purposes, also in the future, both Udacity's dataset and Kitti's will be merged to get even better results. The network used to test the Udacity video has never seen any of that data, and therefore it's interesting to see the results.

#### 3. DetectNet as a mean to detect cars in a highway.

[DetectNet](https://arxiv.org/pdf/1312.6082.pdf) is a derivation from GoogleNet used in its inception to detect numbers in an image, but it's not a simple number classifier network, it takes as input a whole image, and tries to detect all the Street numbers on it. Because it doesn't behave like other past approaches and because it doesn't generate proposals (sections in the image to be classified if car or not) it doesn't need a Sliding window to generate possible frames where cars could be, and it's really robust to scaling and occlusion.

Because of those reasons this network was selected, and a more detailed view can be seen on this next figure:

![alt text][detectnet]

This network was trained during 30 epochs with the weights taken from the street number selection task. It then was fine tunned with 60 more epochs. No augmentation was performed in either of those both stages.

The next image shows the loss in the first 30 epochs, it can be seen that his one is not really accurate, but that's expected from start.

![alt text][detectnet1loss]

The learning rate for training was a polynomial decay, starting with 0.0001 as can be seen in the next figure:

![alt text][detectnet1lr]

After that first run, some fine tunning was done, with 60 more epochs, this used the weights from the past training so a lot of improvement was shown. The loss is shown in the next figure:

![alt text][detectnet2loss]

The learning rate for this one was also a polynomial decay, but with different parameters:

![alt text][detectnet2lr]

#### 4. Results and utilization.

After the training was done (which due to the images length took a lot of time) the results were great, for a test image:

![alt text][source]

The result with bounding boxes drawn was the following:

![alt text][inference]

The advantage of DIGITS, is that it allows visualization at different stages of the inference, so we can clearly see how the neurons are behaving on all stages like this:

![alt text][visual]

Having the last step of the past inference generated a heatmap like layer that is like this:

![alt text][coverage]

These results were satisfactory, since the network had never seen that video before, making it generalized for multiple uses.

#### 5. Video generation.

Because it's Caffe, the following files are needed in order to generate the video:

* `.caffemodel` A Caffemodel file
* `deploy.prototxt` A Caffemodel file
* The input video


And must be called this way:

```
python CarDetection.py snapshot_iter_19778.caffemodel deploy.prototxt project_video.mp4
```
NOTE: Caffe must be installed in the machine for this to work. Instructions on how to build caffe can be found [HERE](https://github.com/NVIDIA/DIGITS/blob/master/docs/BuildCaffe.md)

In the `CarDetection.py` the following functions are to note:
```python
def forward_pass(image, net, transformer, batch_size=None):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)

    Arguments:
    image -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer

    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    if batch_size is None:
        batch_size = 1

    caffe_images = []

    if image.ndim == 2:
        caffe_images.append(image[:,:,np.newaxis])
    else:
        caffe_images.append(image)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        start = time.time()
        output = net.forward()[net.outputs[-1]]
        end = time.time()
        if scores is None:
            scores = np.copy(output)
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images in %f seconds ...' % (len(scores), len(caffe_images), (end - start))

    return scores
```
That does the forward pass of an image through the network and evaluates time.

Once we get the results from the network we use this function to draw the bounding boxes:

```python
def draw_bboxes(image, locations):
    """
    Draws the bounding boxes into an image

    Arguments:
    image -- a single image already resized
    locations -- the location of the bounding boxes
    """
    for left,top,right,bottom,confidence in locations:
        if confidence==0:
            continue
        cv2.rectangle(image,(left,top),(right,bottom),(255,0,0),3)
    #cv2.imwrite('bbox.png',image)#test on a single image
    return image
```
A clustering is done on the outputs to vote on the bounding boxes to be drawn:

```python
def forward(self, bottom, top):
        for i in xrange(self.num_classes):
            data0 = bottom[0].data[:,i:i+1,:,:]
            bbox = cluster(self, data0, bottom[1].data)
            top[i].data[...] = bbox
```
To test the model with the video, please call:

```
python CarDetection.py snapshot_iter_19778.caffemodel deploy.prototxt project_video.mp4
```

---

Videos of the both detections can be seen here:

* This one for the first video:
[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/9a_cF1vNVSw/0.jpg)](http://www.youtube.com/watch?v=9a_cF1vNVSw)

There is some struggle with the white color, but in the future this is expected to be resolved.

* For the harder challenge, it performs even better and detects cars from the other lane also

[![IMAGE ALT TEXT HERE](http://img.youtube.com/vi/ucfXWugvUc0/0.jpg)](http://www.youtube.com/watch?v=ucfXWugvUc0)
---

### Discussion
This project was exciting, specially because I drove me to use something different, so that I can get better results and faster times. The idea of using a complex network, but having it evaluate each image in 0.03 secs average is amazing. I am looking forward to improve this ever further with the help of TensorRT, and also augment the dataset and play with a segmentation network to see how it performs, and if there is improvements for both time and accuracy.

### Acknowledgments
To the Udacity Slack Channel, for all the help and ideas. Also to NVIDIA, who provided me with the TX1 to use in the future deployment of this network.
