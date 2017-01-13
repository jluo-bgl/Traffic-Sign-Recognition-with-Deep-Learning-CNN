## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

This is my implementation of Traffic Sign Recognition Project from [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

I hope this project could help people who is learning deep learning (just like me), it contains lots of pices that 
a beginner will ask.
 1. How to explorer the dataset
 2. How to augment and normalise dataset
 3. During training, how to get the basic information
 - Training Loss/Accuracy, Validation Loss/Accuracy
 - Confusion Matrix
 - Wrong Predictions
 4. Comparison between different models so that give you a feeling how model effect final result
 
I prefer to run code and observe the result rather than just reading document.
This project trying to have most of the code have some how automated test covered.
this is my first python project so that any pull request are welcome. 
 
Whole project are follow Immutable / Compositable Overall Design

To run jupyter notebook, please follow Udacity getting started kit project setup you environment first

1. git clone https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN.git
2. cd Traffic-Sign-Recognition-with-Deep-Learning-CNN
3. wget https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/releases/download/0.1_GPU_Ready/test.p
4. wget https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/releases/download/0.1_GPU_Ready/train.p
5. jupyter-notebook (If aconda environment or your own way to start jupyter notebook)

if you're in aws, make sure open 8888 port in your security group
access public IP address  ip:8888

To setup a AWS GPU instance, use udacity gpu iam is the simplest way

### Data Loading and Validation Data Split
There is a class called "TrafficDataRealFileProviderAutoSplitValidationData" will load data from **"train.p"** and **"test.p"**, also parameters will help us split validation data from train.p
> below code will load from file and split trainging / validation set as 80/20
```python
from traffic.traffic_test_data_provider import TrafficDataRealFileProviderAutoSplitValidationData
original_data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
            split_validation_from_train=True, validation_size=0.20)
```
> below code will load file from the file name you specific
```python
data_provider = TrafficDataRealFileProviderAutoSplitValidationData(
    training_file="your_train_file.p", testing_file="your_test_file.p",
    split_validation_from_train=True, validation_size=0.20)
```

### Data Augment and Preprocessing
Data process procedures are listed in [python file: traffic_data_enhance](../edit/traffic/traffic_data_enhance.py)

first all, it is clearly show that data had a class imbalance.
>highest count of sign names: 'Speed limit (50km/h)' **1800**

>lowest count of sign names: 'Go straight or left', **163**

>that's about **1100%** difference

#### The augment procedure I have
1. Random Rotation between -20 and 20 angles
2. Random Corp Image between 20% and 0.5%
3. Random Brightness and Contrast
After read from 'train.p' file, I will shuffle and split it into 80/20, where 80% used for training, 20% used for validation.
The augment process will only created based on training data set, never touch valiation set.

I found out that even by augment training data with 200% of max class(1800) with method 1 or 2, still dosen't help too much

However, my [Wrong Predictions Picture](https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/raw/master/images/wrong_predictions.png "Wrong Predictions") picture makes me think that the model dosen't work well when picture is too dark or too bright, I think brightness and Contrast will help me.

Brightness and Contrast required lost of computing power, I implemented the code but haven't produce the data yet. will update this later.

### Data Normalisation
Data process procedures are listed in [python file: traffic_data_enhance](../edit/traffic/traffic_data_enhance.py)

Data normalisation helps with training optimisation

I have expirement below normalisation methods
1. No normalisation
2. [0..255] convert to [0..1]
3. [0..255] convert to [-1..1]
4. grayscale
5. tf.image.per_image_standardization (whitening)

No normalisation still gives a good score, but it make the training harder (took longer), as a example,

For LeNet(Basic) network, 
With normalisation 
>EPOCH 50 training loss = 0.015 accuracy = 0.996 Validation loss = 0.083 accuracy = 0.990

Without 
>EPOCH 50 training loss = 0.042 accuracy = 0.989 Validation loss = 0.116 accuracy = 0.983

### Methods and Scores

[LeNet(Basic) Structure in Source Code](https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/blob/master/traffic/lenet.py) 
1. Input 32x32x3(or 1 if grayscale)
2. Filter 5x5x6 output 28x28x6
3. max pool (ksize=[2, 2], strides=[2, 2]) output 14x14x6
4. Filter 5x5x16 output 10x10x16
5. max pool (ksize=[2, 2], strides=[2, 2]) output 5x5x16
6. drop out
7. full connected layer 400 -> 120 -> 84 -> 43

[LeNet(Adv) Structure in Source Code](https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/blob/master/traffic/traffic_lenet_v8_108x200.py) 
1. Input 32x32x3(or 1 if grayscale)
2. Filter 5x5x108 output 28x28x108
3. max pool (ksize=[2, 2], strides=[2, 2]) output 14x14x108
4. Filter 5x5x200 output 10x10x200
5. max pool (ksize=[2, 2], strides=[2, 2]) output 5x5x200
6. drop out
7. full connected layer 5000 -> 1000 -> 200 -> 43

[Inception Structure in Source Code](https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/blob/master/traffic/traffic_net_inception.py)

[Model Comparison Note Book Located Here](https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/blob/master/Traffic_Sign_Classifier_Model_Comparision.ipynb)

| Score | Network     |Augment                  | Normalisation  |Learning Rate|Drop Out|Epoch|
| ------|-------------|-------------------------|----------------|-------------|--------|-----|
| 0.900 | LeNet(Basic)|None                     |None            |0.001        |None    |80   |
| 0.926 | LeNet(Basic)|None                     |None            |0.001        |0.5     |80   |
| 0.938 | LeNet(Basic)|None                     |standardization |0.001        |None    |80   |
| 0.956 | LeNet(Basic)|None                     |standardization |0.001        |0.5     |80   |
| 0.951 | LeNet(Basic)|Rotate                   |standardization |0.001        |0.5     |80   |
| 0.954 | LeNet(Basic)|Rotate+Brighten+Contrast |standardization |0.001        |0.5     |80   |
| 0.954 | LeNet(Adv)  |Rotate+Brighten+Contrast |standardization |0.001        |0.5     |80   |
| 0.958 | LeNet(Adv)  |None                     |standardization |0.001        |0.3     |100  |
| 0.954 | Inception   |None                     |standardization |0.0005       |0.3     |80   |

Precetage of valiation data?
20 / 30 ?
or 200 pre class?

Grayscale or Not?

Image normalise -1..1, or 0..1 or 0.1 to 1

Variable Stddev, 0.1 or 1 or 2 or ...
mean, 0 or ...

batch size ?

epoch?



### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Paper: [Traffic Sign Recognition with Multi-Scale Convolutional Networks](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)


Enhanced Data
10 / 20 /30 Precent Validation Data
GrayScale or not

Visualise Training Loss

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project and start the notebook.
```
git clone https://github.com/udacity/CarND-Traffic-Signs
cd CarND-Traffic-Signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```
3. Follow the instructions in the `Traffic_Signs_Recognition.ipynb` notebook.

