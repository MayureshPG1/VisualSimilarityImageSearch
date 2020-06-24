#### This project is solution to visual similarity based image search

### Dataset Provided

Fashion dataset taken from myntra.com (Indian e-commerce website)

[Large Dataset 15GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Small Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)


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

 - Now, instead of finding first K matches with test image. Can we precompute the distances.

 - Thus, we now compute cosine distance of every single image in database with every other image and store it as a table.

 - So at runtime; we just have to load the table find top 10 matchings by simply traicing the table.

### Optimization Results

For a batch of 100 test images following are the average execution time for each version of algorithm.

| Algorithm        | Avg ext time           | gain  |
| ------------- |:-------------:| -----:|
| V_1.0      | 8.3 Seconds | - |
| V_2.0      | 9 millisecond      |  922x faster |
| V_3.0 | 4 millisecond      |    2.25x faster |

### Literature reffered

  - My implementation is based on [following paper](http://cs231n.stanford.edu/reports/2015/pdfs/nealk_final_report.pdf)
  - [Good overview](https://medium.com/de-bijenkorf-techblog/image-vector-representations-an-overview-of-ways-to-search-visually-similar-images-3f5729e72d07) of though process behind arriving at using pre-trained CNN based model.
  - [Similar approach](https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html) using ResNet. 
  - Visual similarity detection at [Pinterest](https://labs.pinterest.com/user/themes/pin_labs/assets/paper/visual_search_at_pinterest.pdf)
    - Their entire network is quite complicated. But, to find visual similarity they use a similar approach.
        > Training a full CNN to learn a
      good representation can be time-consuming and requires a
      very large corpus of data. We apply transfer learning to
      our model by retaining the low-level visual representations
      from models trained for other computer vision tasks.
  - Did not used [this paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_Cross-Domain_Image_Retrieval_ICCV_2015_paper.pdf) as it uses labelling data along with visual similarity
    - But, gained some insights about fashion datasets and visual analysis related to clothing
  - [This paper's](http://acberg.com/papers/wheretobuyit2015iccv.pdf) problem domain is different. They are figuring cloths for shop and general photos.
    - But, They demonstrated that cosine based distances work the bast. So, I ended up using them.
  - Did not used [this approach](https://blog.griddynamics.com/reverse-image-search-with-convolutional-neural-networks/). This one requires 3 input images query, positive and negative.
    - Thought that asking 3 input images to the user is not intuitive.
    
### Why Not GAN based approach

- Basically there are 2 steps in solving visual similarity problem
  1. calculating feature vectors for database images
  2. Finding closest images to test images with use of some kind of distance. (Euclidian, cosine etc)
 - The GAN network solves first step by using very complicated network.
  - i.e. you've to train two CNN networks
  - These 2 networks are in feedback loop with each other
  - Eventually going through this network we produce a similar feature vector
  - It is not proven anywhere in the paper that this GAN feature vector is better than feature vector calculated by other methods.
 - Most importently in terms of precision; according to paper itself.
  - GAN approach achieves 0.84 precision
  - But at the same time other methods give very close results
    - VGG10(0.81), ResNet101(0.8), ResNet152(0.82)
    - *Thus with so much added complexity GAN results are not significantly better.*
  - Also, it's obivious that difference in precision of different CNN models will very based input dataset
    - As these models are trained on different dataset. So it's always a guess how generic filters are of a specific CNN model for your dataset.
    - You can always train your CNN model with your dataset (in our case fashion).
    - Or get a pre trained CNN model on a similar dataset. i.e. FashionNet.
  - There are practical considerations too
    - From the paper it looks like you must train the GAN network with your dataset.
      - Which means days to weeks of training effort in multiple epochs
    - No transfer learning
      - Meaning we cant do quick prototyping to evaluate the algorithm.
  - Moreover the precision results claimed are not on some standard dataset. Dataset is curated by authors.
    - If there was a standard dataset (such as ImageNet challenge etc) then the results can be given more weightage.
    - Also, results are on one specific dataset. It's note proven that GAN method works best on all different kind of dataset.
  - While going through methods used by many commercial companies which has commercially solved the visual similarity problem; I noticed that most of them have used pre-trained CNN networks. The only question was which one.
    
 Thus the additional complexity of implementation and being not sure about precision results I decided not to implement GAN paper.

### Notes

  - It's much better to craft this problem as finding closest image from given dataset.
    - This allows hugh performance gains. Also, it's more practical problem. i.e. User in on an e-commerce website. User searches for white shirt with design and below my algorithm suggest 10 best matching T-shirts.
    - Also, all images from dataset is considered as input images. There is no training, validation and test data kind of problem modelling.
We are using pre-trained model and getting image's feature vectors from that model.
  - I was also unable to load 15GB database. So, I used small dataset.
  - For small dataset also I was able to upto max 10K images. Trying further made the system crash.
  - My laptop has no GPU. Will need a GPU machine to scale up the results.
### Future work and Optimizations

 - Use a high end GPU or hire cloud GPU's to speed up the training process.
 - We can calculate the precision of out prediction using labelled data. i.e. for white-male-design-shirt all outputs of similar images must match the label.
  - But, some papers do mention that visual similarity as a subjective problem and using labels may not be best answer.
 - Also as size of data increase (e.g. 100K images to compare). We need to recheck if V_2.0 approach is better or V_3.0 approach is better.
  
