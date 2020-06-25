## Visual Similarity Based Image Search

--------------

### Dataset Provided

Fashion dataset is taken from myntra.com (Indian e-commerce website)

[Large Dataset 15GB](https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset)

[Small Dataset](https://www.kaggle.com/paramaggarwal/fashion-product-images-small)

--------------

### Problem Statement

Design and implement a system that computes visual similarity. Think of a visual search engine that provides you with similar
searches when given an image input.

------------------

### Problem Definition

Here is my thought process:

The problem to be solved is searching images based on visual similarity. The dataset provides labeled data of all images but I am not going to use that for finding similar images. If we use labels, this would become a Natural Language Processing kind of problem.

I also need to find 'k' number of most similar images. Suggesting its very likely that problem will be modeled as solving K Nearest Neighbours(KNN).

Also to find visual similarity, we need an invariant feature vector. (Invariant to changes light, size, orientation, color, etc.)

The feature vector is an abstract representation of the image. This vector encodes image data in a more compact and feature-rich manner.

One of the best ways to get a very good feature vector is to use the last layer of pre-trained Convolutional Neural Network.

In CNN's, initial layers represent simple filters detecting edges and curves, etc. And at the last layer, we get the final abstract response of an image.

I've decided to use VGG16 CNN trained on ImageNet. As its widely used and its weights are available in Keras.

I've used model trained on ImageNet as its a very wide dataset with 1000 output classes. Thus the filters trained on it would be generic enough that our fashion images should produce very good feature vectors. (If not we've to train VGG with our dataset or use pre-trained datasets such as FashionNet.)

So, I am essentially planning to use transfer learning from a pre-trained network. Get image feature vectors and use a KNN like classifier to find K nearest similar images to a given test image.

--------------

### Tech stack

I've used the following tech stack:
- Tensorflow (Widely used. Provides good deep learning libraries)
- Keras (Provides good high-level abstractions API and a lot of pre-trained networks available)
- Python (Very good with 2d,3d data processing and a lot of other libraries available)
- Numpy, Pandas, PIL, sklearn, etc (For performing low-level tasks)
- Jupyter Notebook. (Easy to present and visualize code)

--------------

### Design

Following is a pseudo algorithm:

- Load VGG16 model trained on ImageNet.
- Get the last layer of the model.
- Load images from the database.
- Preprocess images and make them compatible with the input layer of VGG16.
- Pass each image as an input to the VGG model and store the response of the last layer as an image's feature vector.
- Pass this feature vectors array as an input to the KNN model. Use cosine distance.
- Load the Test image.
- Preprocess the Test image.
- Load the VGG16 model again and get test image feature vectors.
- Pass this test feature vector to the KNN model. It returns the first 10 best matching feature vectors.
- Display test and matching images with scores.

--------------

### Code Optimizations

**V_1.0** 
Implements above mentioned basic algorithm.

**V_2.0:**
*Following are the problems and modification in V_2.0:*

- To execute the code we have to train and extract the feature vectors each time. This takes a long time to check results.
  - Now let's divide to code into training and testing part
- Let's save the feature vectors and KNN model to the local disk at the end of training and reload it back at the time of testing
  - This will also reduce the number of variables that need to be in memory all the time.
 - As feature vectors are saved, no need to calculate the test image feature vector at the time of execution. We can directly fetch the feature vector from the saved array.
 
 **V_3.0**

*Let's make test execution even faster at test time*

 - Now, instead of finding first K matches with the test image. Can we precompute distances?

 - So, we now compute the cosine distance of every single image in the database with every other image and store it as a table.

 - Now at runtime; we just have to load the table find top 10 matchings by simply tracing the table.

--------------

### Optimization Results

For a batch of 100 test images following are the average execution time for each version of the algorithm.
(Results will vary based on input datasize)

| Algorithm        | Avg ext time           | gain  |
| ------------- |:-------------:| -----:|
| V_1.0      | 8.3 Seconds | - |
| V_2.0      | 9 millisecond      |  922x faster |
| V_3.0 | 4 millisecond      |    2.25x faster |

--------------

### Literature reffered

  - My implementation is based on [following paper](http://cs231n.stanford.edu/reports/2015/pdfs/nealk_final_report.pdf)
  - [Good overview](https://medium.com/de-bijenkorf-techblog/image-vector-representations-an-overview-of-ways-to-search-visually-similar-images-3f5729e72d07) of though process behind arriving at using pre-trained CNN based model.
  - [Similar approach](https://www.oreilly.com/library/view/practical-deep-learning/9781492034858/ch04.html) using ResNet. 
  - Visual similarity detection at [Pinterest](https://labs.pinterest.com/user/themes/pin_labs/assets/paper/visual_search_at_pinterest.pdf)
    - Their entire network is quite complicated. But, to find visual similarity they use a similar approach.
        > Training a full CNN to learn a
      good representation can be time-consuming and requires a
      a very large corpus of data. We apply transfer learning to
      our model by retaining the low-level visual representations
      from models trained for other computer vision tasks.
  - Did not used [this paper](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Huang_Cross-Domain_Image_Retrieval_ICCV_2015_paper.pdf) as it uses labeling data along with the visual similarity
    - But, gained some insights about fashion datasets and visual analysis related to clothing
  - [This paper's](http://acberg.com/papers/wheretobuyit2015iccv.pdf) problem domain is different. They are figuring cloths for shop and general photos.
    - But, they demonstrated that cosine based distances work the bast. So, I ended up using them.
  - Did not used [this approach](https://blog.griddynamics.com/reverse-image-search-with-convolutional-neural-networks/). This one requires 3 input images query, positive and negative.
    - I thought that asking 3 input images to the user is not intuitive.
    
--------------

### Why Not GAN based approach

[Reference Paper](https://www.researchgate.net/publication/336728075_A_Visual_Similarity_Recommendation_System_using_Generative_Adversarial_Networks)
- In general, there are 2 steps in solving visual similarity problem
  1. calculating feature vectors for database images
  2. Finding the closest images to test images with the use of some kind of distance. (Euclidian, cosine, etc)
 - The GAN network solves the first step by using a very complicated network.
  - i.e. you've to train two CNN networks
  - These 2 networks are in a feedback loop with each other
  - Eventually going through this network we produce a similar feature vector
  - It is not proven anywhere in the paper that this GAN feature vector is better than the feature vector calculated by other methods.
 - Most importantly in terms of precision; according to the paper itself:
  - GAN approach achieves 0.84 precision
  - But at the same time, other methods give very close results
    - VGG10(0.81), ResNet101(0.8), ResNet152(0.82)
    - *Thus with so much added complexity GAN results are not significantly better.*
  - Also, the difference in the precision of different CNN models will vary based input dataset
    - Basically a pre-trained model is trained on specific dataset. Model 'learns' filters based on given dataset. If different dataset is passed to same CNN model results may vary. i.e. we don't know how generic the filters of pre-trained models are.
    - You can always train your CNN model with your dataset (in our case fashion).
    - Or get a pre-trained CNN model on a similar dataset. i.e. FashionNet.
  - There are practical considerations too
    - From the paper, it looks like you must train the GAN network with your dataset.
      - Which means days to weeks of training effort in multiple epochs
    - No transfer learning
      - Meaning we cant do quick prototyping to evaluate the algorithm.
  - Moreover, the precision results claimed are not on some standard dataset. Dataset is curated by authors.
    - If there was a standard dataset (such as ImageNet challenge etc) then the results can be given more weightage.
    - Also, results are on one specific dataset. It's not proven that the GAN method works best on all different kinds of datasets.
  - While going through methods used by many commercial companies; I noticed that most of them have used pre-trained CNN networks.
    
 Thus the additional complexity of implementation and being not sure about precision results I decided not to implement GAN paper.

--------------

### Notes

  - It's much better to craft this problem as finding the closest image from the given dataset. i.e. Test image is from dataset only.
    - This allows for huge performance gains. 
    - Also, it's a more practical formulation. i.e. User in already looking at a shirt on an e-commerce website. The image of shirt is from database only and now we show similar matching images from database.
    - Also, all images from the dataset are considered as input images. There is no training, validation, and test data kind of problem modeling.
We are using a pre-trained model and getting the image's feature vectors from that model.
  - I was also unable to load a 15GB database. So, I used a small dataset.
  - For the small dataset, I was able to do up to max 10K images. Trying further made the system crash.
  - My laptop has no GPU. We will need a GPU machine to scale up the results.

--------------

### Future work and Optimizations

I completed what I could in 4 days. But following are the optimizations I could think of.

 - We can calculate the precision of our predictions using labeled data.
  - But, some papers do mention that visual similarity as a subjective problem and using labels may not be the best answer.
 - Also as the size of data increase (e.g. 15GB data). We need to recheck if the V_2.0 approach is better or the V_3.0 approach is better.
 - Parallelism can be used in V_2.0 for large dataset. Instead of finding KNN matches with 100k images, we can split the data and find best matches in small subsets of data.
 - The dataframe/matrix generated in V_3.0 has a lot of redunduncies. It is calculating cosine distance of all images with each other. So, dataframe[1][5] is same as dataframe[5][1]. i.e. distance between image 1 & 5. 
  - Also, if we know we only need first 'K' matches then we can sort the matrix columnwise and keep first 10 elements with lowest scores only.
 - Use a high-end GPU or hire cloud GPU's to speed up the training process.

--------------

### How to Run

I've uploaded 3 separate versions of algorithms in jupyter notebook format (ipynb)

If you open these notebook directly you'll see all the results in them below each code block.
You can open them in Jupyter Notebook and execute the code blockwise.

I could not upload the saved data for V_2.0 and V_3.0 due to size limit on github. So, I've uploaded them [here](https://1drv.ms/u/s!AiAKI2YLMbz_h6Vz97oijr_iiUNFhw?e=tFHIii) along with dataset.

Download the data and keep it in the same folder as ipynb notebooks. The code will pick them up. In which case you dont need to run the training code. Run the test code directly, it will load the saved data and run the algorithm.

Your system should have following:
 - Tensorflow
 - python 3.7
 - numpy
 - pandas
 - sklearn
 - PIL
 - matplotlib

  Regarding docker:
  
  - I've used Anaconda for creating my tensorflow environment.
  - Docker and Anaconda are not really compatible and there are lot of hacks to get them working together.
  - In retrospect if I eventually had to give an Docker image I should not have used Anaconda.
  - Working on getting Docker image out of Anaconda environment.
  - If I could solve this will send Docker image soon.
