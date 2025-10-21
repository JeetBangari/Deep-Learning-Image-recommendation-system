# Deep-Learning-Image-recommendation-system
This repository shows the program for image recommendation system using deep learning.

We are going to use the strategy or approach of Triplet Loss method in ML. In Triplet Loss Method we generate 3 pair imges known as (A,P,N) A-Anchor image , P-Positive image 
and N-Negative image. Anchor image is the input image from the user. Positive image is the Image similar to the anchor image(i.e; From same class as Anchor) and Negative image means from a different class from anchor. The distance between the Anchor and Positive and Anchor and NEgative is measured(Euclidian Distance) then adding it on a margin (generally 0.2 is considered better). The condition for similarity search is that D(A,P) < D(A,N).

The project is divided into 5 stages:-
STAGE 1:-Data Curation and Preparation. This stage has 2 steps which are listed as below.
Step 1:-In this stage we collected data online from google open images the link for the same is https://storage.googleapis.com/openimages/web/index.html. After collecting it we organised it in our hard disk to use it in project.
Step 2:-Implement the triplet mining strategy which includes creating a triplet generator function which will generate our (A,P,N) triplet as a tuple to feed to our Model.

STAGE 2:-Model Architure and Transfer Learning. This stage has 3 steps.
Step 1:- Choosing a pretrained model of CNN with pre-trained weights. For this project we have used ResNet50 pre-trained CNN Model which was trained by ImageNet.
Step 2:-We will remove the final lyer from this pre-trained model and add our own dense layer to learn about our own image pattern and adjust weights of our model accordingly which will be followed by L2NORMALIZATION to the base CNN to restrict the output within (0,1) to easy our further model calculations.
Step 3:-In this step we will define and compile our triplet model by passing the main triplet function that is D(A,P) - D(A,N) + Margin.

STAGE 3:- Training And Evaluation. This stage has 2 steps.
Step 1:-In this step we will feed the triplet generated from our triplet generator function to the model built in stage 2. We will be feeding te model with data batch wise. It will be difficult for the model to learn all the patterns and similarity in a single feed of data. Hence to make it learn soomthly we will feed it batch wise which will be fed to the model in each iteration. For our project our batch size is 32. This means that in one iteration we will feed it 32 triplets of images i.e; 32 Anchor images , 32 Positive images , 32 Negative images which makes 96 images total in one iteration.
Step 2:- After each epoch we will be evaluating the model's performance so that we don't undertrain it as well as don't overtrain it.

STAGE 4:-Feature Extraction and Indexing. This stage has 2 steps.
Step 1:-We will pass our complete dataset to the model and extract similar features from the images to help our model select similar images.
Step 2:- We will initiate and fit the Nearest Neighbours(NN) Model.

STAGE 5:-Deployment and Testing. This stage has 2 steps.
Step 1:- Create a website using streamlit library of python and execute the following the code.
Step 2:- Use the unseen images to test the model.

The project created here is a small scale deep learning model with only 50 Epoches and 1000 images of dataset out of which only 700 images were used to train the data and 50 images were used for validation task. Hence the Model way vary form high accuracy.

Through this project we are only tring to demonstrate one very basic deep learning model aiming to cover most of it's basics.

For this project we have used Tensorflow(Keras integrated API), Numpy, OS , Pillow , Streamlit , Random , Pickel libraries.

This was an full overview of te project.
