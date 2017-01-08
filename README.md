## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
### Overview

deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

Trying to have most of the code unit test covered, but this is my first python project so that any pull request are welcome.

Immutable / Compositable Overall Design

git clone https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN.git
cd Traffic-Sign-Recognition-with-Deep-Learning-CNN
wget https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/releases/download/0.1_GPU_Ready/test.p
wget https://github.com/JamesLuoau/Traffic-Sign-Recognition-with-Deep-Learning-CNN/releases/download/0.1_GPU_Ready/train.p
jupyter-notebook

if you're in aws, make sure open 8888 port in your security group
access public IP address  ip:8888

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

