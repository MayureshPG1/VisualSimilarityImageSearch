#### This project is solution to visual similarity based image search

### Dataset Provided

Fashion dataset taken from myntra.com (Indian e-commerce website)

https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset

https://www.kaggle.com/paramaggarwal/fashion-product-images-small

### Problem Statement

Design and implement a system that computes visual similarity. Think of a visual search engine that provides you with similar
searches when given an image input.

### Problem Definition

Here is my thought process:

The problem to be solved is of searching images based on visual similarity. The dataset provides labeled data of all images but I am not going to use that for finding similar images. If we use labels, this would become a Natural Language Processing kind of problem.

I also need to find 'k' number of most similar images. Suggesting its very likely that problem will be modelled as solving K Nearest Neighbours(KNN).

Also to find visual similarity, we need an invarient feature vector. (Invarient to changes light, size, orientation, color etc.)

Feature vector is an abstract representation of the image. This vector encode image data in more compact and feature rich manner.

One of the best to get a very good feature vector is to use last layers of pre-trained Convolutional Nueral Network.

In CNN's, initial layers represent simple filters detecting edges and curves etc. And at the last layer we get final abstract respose of an image to all the filters in all layers.

I've decided to use VGG16 CNN trained on ImageNet. As its widely used and it's weights are available in Keras.

I've used model trained on ImageNet as its a very wide dataset with 1000 output classes. Thus the filters trained on it would be generice enought that our fashion images should produce as very good output response. (If not we've to train VGG with our dataset or use pretrained datasets such as FashionNet.

So, I am essentially planning to use transfer learning from an pretrained network. Get image feature vectors and use a KNN like classifier to find K nearest similar images to given test image.

### Tech stack

I've used following tech stack:
- Tensorflow (Widely used. Provides good deep learning libraries)
- Keras (Provides good high level abstractions API and a lot of pre trained networks available)
- Python (Very good with 2d,3d data processing and lot of other libraries available)
- Numpy, Pandas, PIL, sklearn etc (For performing low level tasks)
- Jupyter Notebook. (Easy to present and visualize code)

### Design

Following is psudo algorithm:

- Load VGG16 model trained on ImageNet.
- Get model upto last layer.
- Load images from database.
- Preprocess them and make them compatible with input layer of VGG16.
- Pass each image as an input to VGG model and store response of the last layer as image's feature vector.
- Give this feature vectors array as an input to KNN model. Use cosine distance.
- Load the Test image.
- Preprocess the Test image.
- Load VGG16 model again and get test image feature vectors.
- Pass this test feature vector to KNN model. It returns first 10 best matching feature vectors.
- Display test and matching images with scores.

### Optimizations

**V_1.0** 
Implements above mentioned basic algorithm.

**V_2.0:**
*Following are the problems and modification in V_2.0:*

- To excute the code we have to train and extract the feature vectors each time. This takes long time to check results
  - Now lets divide to code into training and testing part
- Lets save the feature vectors and knn model to the local disk at the end of training and reload it back at time of testing
  - This will also reduce number of variables which needs to be in the memory all the time.
 - As feature vectors are saved, no need to calculate the test image feature vector at the time of execution. We can directly fetch feature vector from saved array.
 
 **V_3.0**

*Let's make test execution even more faster at test time*

Now, instead of finding first K matches with test image. Can we precompute the distances.

Thus, we now compute cosine distance of every single image in database with every other image and store it as a table.

So at runtime; we just have to load the table find top 10 matching by simply traicing the table.

### Optimization Results

For a batch of 100 test images following are the average execution time for each version of algorithm.

| Algorithm        | Avg ext time           | gain  |
| ------------- |:-------------:| -----:|
| V_1.0      | 8.3 Seconds | - |
| V_2.0      | 9 millisecond      |  922x faster |
| V_3.0 | 4 millisecond      |    2.25x faster |



### Notes

It's much better to craft this problem as finding closest image from given dataset.

This allows hugh performance gains. Also, it's more practical problem. i.e. I'm an e-commerce website. I search for white shirt with design and below my algorithm suggest 10 best matching T-shirts.

Also, all images from dataset is considered as input images. There is no training, validation and test that in this problem modelling.
We are using pre-trained model and getting image's feature vectors from that model.
