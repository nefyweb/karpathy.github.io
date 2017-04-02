<div>
<style>
table{
    border-collapse: collapse;
    border-spacing: 0;
    border:1px solid #000000;
}

th{
    border:1px solid #000000;
}

td{
    border:1px solid #000000;
}

th, td {
    padding: 7px;
    text-align: left;
tr:hover {background-color: #e7f6fd}
}

th {
    background-color: #e7f6fd;
    color: #606060 ;
}

img.center {
    display: block;
    margin: 0 auto;
    padding: 5px 5px 5px 5px;
}

img {
    display: block;
    margin: 0 auto;
    padding: 5px 5px 5px 5px;
}
</style>
</div>
#### Deep Leaf (DL) Business Applications

The DL model is ensemble spread over two types of data, textual and image data for the purpose of classification. Image classificaton has may applications in business and indepth financial speculation and analysis. 

The textual data here comes from sensory measurement information as it is presented in a quantative fashion. The data gatherers therefore also took other type of measurement. 

#### Finance
There are countless opportunities for more speculative financial predictions and also for indepth analysis that can be invaluable for  choosing amongst stocks. 

- Models trained on aerial photography and geospatial data to identify the number of cars and model of car at an automakers factory to get fast access to location specific information. 
- Imagery to quickly identify the quality of products (material) from public companies and sensory inputs relating to the texture and the weight of the product. 
- Imagery inspecting airline carriers and type of carriers to get an indication of the tourist demand in certain areas. The inclusion of radio or other signal information to identify the trajectory of the flight and the potential destination.

#### Business 
Similar to the finance applications, there are thousands of opportunities to improve your internal business management.   

- Conveyer belt classification for quality management, to identify whether the product is up to standards, this can include weight, size smell and texture measurements all of which can be fed into a mode simlar to DL. A real world example of this is a farmer using classification models to sort through his cucumber harvest. 
- Imagery for inventory management, where inventory can not easily be quantatively stated, such as with piles of cement, water levels or other commoditee type goods.
 
---

#### Model Description 

The first model we use is **sequential** from the **Keras** **API** and it is an **MLP** model for which I specified certain paramaters.

| Terminology        | Description         | 
| ------------- |-------------| 
|Keras     | Is a high-level neural network library | 
|Sequential     | One of the two types of Keras Models | 
|API     | A set of functions and procedures that allow the creation of applications which access the features or data of an operating system, application, or other service  | 
|MLP     | Multiple Layer Perceptron - Same as ANN (Artificial Neural Network), encompasses all deep learning constructs. | 
|Parameters     | The adjustable variables that can be tuned by algorithms or by the developer to build a better model (accuracy, generalisability) |




In this scenario, I made use of a standard MLP model to investigate the textual features. This dataset includes pre-extracted feauture information but also near-raw images that can be used to self-extract features. In this example I would create an ensable of a simple **ANN** (MLP for the textual data) and a convolutional neural network (**CNN**) to extract features from the images.  

**Convolutional Neural Networks** are ANNs (i.e. MLPs) with a special structure in which the connectivity patterns of neurons is inpired by the *organisation of the visual cortex*, hence the good results with image recognition.

An example of a deep convolutional network:
(CNN  with many layers are called deep, thus DNN)




<p align="center">
<img src="/assets/leaf/DNN.png" alt="Drawing" style="width: 350px;"/>
</p>

| CNN |
|-------------|
|**Space or Time:** CNNs have repetitive blocks of neurons that are applied across space (for images) or time (for audio signals etc) - Layers from left to right can therefore represent an image space or a time space.|
|**Neurons:** For images, these blocks of neurons can be interpreted as 2D convolutional kernels, repeatedly applied over each patch of the image.|
|**Kernels**: A kernel is a *similarity* function, instead of taking an image, computing and vectorising its features and feeding the feature into a learning algorithm, you only define a single kernel function to compute the *similarity* between images. You provide this kernel that creates a classifier|
|**Dot Products:** The perceptron formulation does not work with kernels, it works with feature vectors, so why do we use kernels. Because every kernel function can be expressed a dot product in a feature space. 2. And a lot of MLA (Machine Learning Algorithms) can be expressed entirely as dot products.|
|**Perceptron:** It is the most basic form of an activation function and is simple a binary function that has only two possible results. The orange output can be 1 or 0, based on the activation functions as applied to the yellow inputs. The *activation function* defines the output of that node given an input or set of inputs. <img src="/assets/leaf/Perceptron.png" alt="Drawing" style="width: 180px;"/> |
|**Components:** There are four main operations in the CNN (Also called ConvNet model, 1. *Convolution*, 2. *Non Linearity (ReLU)*, 3. *Pooling or Sub Sampling*, 4. *Classification (i.e. Fully Connected Layer)*.|

| 1. Convolution |
|-------------|
|**Images:** Digitized images are a matrix of pixel values. See how we can simply turn this 8 into a number representing the intensity of black (the black channel)   <img src="/assets/leaf/8.gif" alt="Drawing" style="width: 180px; padding:5px;"/> An image from a standard digital camera will have three channels – red, green and blue – you can imagine those as three 2d-matrices stacked over each other (one for each color). We give each a pixel values in the range 0 to 255.|
|**Convolution:** The primary purpose of Convolution in case of a CNN is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data.<br> <br> A simple example where intensities are numerically reprenented in 1s and 0s instead of a range from 0-255, the convolutional algorithm moves around and computes a new matrix  <img src="/assets/leaf/conv.gif" alt="Drawing" style="width: 180px;"/>|
|**Algorithm:**  We slide the orange matrix over our original image (green) by 1 pixel (also called ‘stride’) and for every position, we compute element wise multiplication (between the two matrices) and add the multiplication outputs to get the final integer which forms a single element of the output matrix (pink).|
|The reason for choosing this special structure, is to exploit spatial or temporal invariance in recognition. For instance, a "dog" or a "car" may appear anywhere in the image. If we were to learn independent weights at each spatial or temporal location, it would take orders of magnitude more training data to train such an MLP. |
|**Terminology:** In CNN terminology, the 3×3 matrix is called a ‘filter‘ or ‘kernel’ or ‘feature detector’ and the matrix formed by sliding the filter over the image and computing the dot product is called the ‘Convolved Feature’ or ‘Activation Map’ or the ‘Feature Map‘. It is important to note that filters acts as feature detectors from the original input image.|
|**Manipulate:** Using various other computational functions we can create those all to well known operations/features such as sharpening, blurring or detecting the edges of an image. We do this by defining our filter/kernel matrix. So different filters/kernels can detect different features from the an image. GIMP, the GNU image manipulation tool, uses 5x5 of 3x4 matrices such as what we do.|
|**Features:** So we can create an operationg by using preformatted 3x3 matrices (filters) to create image outputs, that we can use a features, there are the 'convolved' images <img src="/assets/leaf/convolved.png" alt="Drawing" style="width: 300px;"/>|
|**Variety:** If, for each image you have such a list of convolved images, it is worth understanding that the classifier (classification algorithm) will now find it easier to identify similar images|
|**Filters:** In practice, a CNN learns the values of these filters on its own during the training process (although we still need to specify parameters such as number of filters, filter size, architecture of the network etc. before the training process). The more number of filters we have, the more image features get extracted and the better our network becomes at recognizing patterns in unseen images. This therefore means that the filters are not preformatted or trained such as in GIMP.|
|**Paramaters:** The size of the Feature Map (Convolved Feature) is controlled by three parameters that we need to decide before the convolution step is performed: *Depth, Stride, Zero-Padding*|
|**Depth:** This is simply the number of filters we use, a specification of 7 depth would mean 7 filters, an example would be the 7 filters produced in the imge above. The more debth the more filters, the more feature maps (features)|
|**Stride:** Stride is the number of pixels by which we slide our filter matrix over the input matrix. When the stride is 1 then we move the filters one pixel at a time. When the stride is 2, then the filters jump 2 pixels at a time as we slide them around. Having a larger stride will produce smaller feature maps.|
|**Zero-padding:** Sometimes, it is convenient to pad the input matrix with zeros around the border, so that we can apply the filter to bordering elements of our input image matrix. A nice feature of zero padding is that it allows us to control the size of the feature maps. Adding zero-padding is also called wide convolution, and not using zero-padding would be a narrow convolution. Have a look at the gif if you don't comprehend the function of inducing zero-padding. |

| 2. ReLU |
|-------------|
|**Non Linearity:** After every Convolutional operation and additional operation called ReLU is performed, it is a non-linear operation that stands for Rectified Linear Unit. It is a simple trnaformation making sure that the output is not higher than the input or lower than zero. <br> `OUTPUT = MAX(ZERO, INPUT)`| 
|**Generalisability:** ReLU is an element wise operation (applied per pixel) and replaces all negative pixel values in the feature map by zero. Most of the real world data is non-linear, so we have to remove the linearity created by the convolution network. Visuaaly seen, it removes the negative black values. This allows for better generalisability. <img src="/assets/leaf/relu.png" alt="Drawing" style="width: 500px;"/> Other non linear functions such as tanh or sigmoid can also be used instead of ReLU, but ReLU has been found to perform better in most situations.| 


| 3. Pooling/Sub-sampling  |
|-------------|
|**Dimensionality Reduction:** Spatial Pooling (also called subsampling or downsampling) reduces the dimensionality of each feature map but retains the most important information.| 
|**Types:** Spatial Pooling can be of different types: Max, Average, Sum etc. In case of Max Pooling, we define a spatial neighborhood (for example, a 2×2 window) and take the largest element from the rectified feature map within that window. Instead of taking the largest element we could also take the average (Average Pooling) or sum of all elements in that window. In practice, Max Pooling has been shown to work better. See below image for max pooling in action. <img src="/assets/leaf/max.png" alt="Drawing" style="width: 350px;"/> |
|**Sofar:** We have images that we vectorize into matrices, then paramater specifications affecting the size of the feature map, those specifications are fed into to convolved algorithms spitting out feature maps, then ReLU or other non-linear functions smooths or rectifies the features maps (Size of the map depending on the paramater specifications (Depth, Stride, Padding)), then we decrease the dimensions with a pooling operation. The function of Pooling is to progressively reduce the spatial size of the input representation|  
|**Logic Behind Pooling:** <br>- Makes the input representations (feature dimension) smaller and more manageable <br> - Reduces the number of parameters and computations in the network, therefore, controlling overfitting <br> - Makes the network invariant to small transformations, distortions and translations in the input image (a small distortion in input will not change the output of Pooling – since we take the maximum / average value in a local neighborhood). <br> - Helps us arrive at an almost scale invariant representation of our image (the exact term is “equivariant”). This is very powerful since we can detect objects in an image no matter where they are located <br> |
|**Convolutional Layers:** All the above is only simple representations of a the basic building blocks of a CNN. The CNN can have two or more sets of *Convolution, ReLU and Pooling Layers*. The next layer simply maps features based on the output of the previous layer|
|**The Theory:** The theory is that when you use this approach the layers together extract the useful features from the images and introduce non-linearity in our network, reduces dimensions while making features quivariant to scale and translation. The output of the last pooling Layer acts as an input to the *Fully Connected Layer*|

| 4. Fully Connected Layer  |
|-------------|
|**MLP:** The Fully Connected layer is a traditional Multi Layer Perceptron (ANN) that uses a softmax activation function in the output layer (other classifiers like SVM can also be used, but will stick to softmax in this post). The term “Fully Connected” implies that every neuron in the previous layer is connected to every neuron on the next layer.|
|**High Level Features (HLF):** The output from the convolutional and pooling layers represent high-level features of the input image. The purpose of the Fully Connected layer is to use these features for classifying the input image into various classes based on the training dataset.|
|**Softmax Sum to One:** The sum of output probabilities from the Fully Connected Layer is 1. This is ensured by using the Softmax as the activation function in the output layer of the Fully Connected Layer. The Softmax function takes a vector of arbitrary real-valued scores and squashes it to a vector of values between zero and one that sum to one. A crude example of the process:   <img src="/assets/leaf/crude.png" alt="Drawing" style="width: 400px;"/>|

| Concluding CNN  |
|-------------|
|**Together:** As discussed above, the **Convolution** + **Pooling layers** act as *Feature Extractors* from the input image while **Fully Connected layer** acts as a *classifier.*|
|**Deep:** In general, the more convolution steps we have, the more complicated features our network will be able to learn to recognize. For example, in Image Classification a ConvNet may learn to detect edges from raw pixels in the first layer, then use the edges to detect simple shapes in the second layer, and then use these shapes to deter higher-level features, such as facial shapes in higher layers|
|**Visualise:** Lastly to use a visual way of understand the process, follow this link: http://scs.ryerson.ca/~aharley/vis/conv/. <br>The below image is an example of an '8', in this scenario a leaf, that has to be classified by species using a learning algorithm.|

----------

<img src="/assets/leaf/1286.jpg" alt="Drawing" style="width: 100%;"/>

---

```python
# Data Manipulation Libraries
import numpy as np
import pandas as pd

# Scikitlearn Libraries
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit

# Keras Libraries
from keras.utils.np_utils import to_categorical
from keras.preprocessing.image import img_to_array, load_img

# Custom 
# Variable decleration. 

# If ever the randomizer gets called on a numpy object, we specify...
#...a seed so that we can generate the same results in the future
np.random.seed(2017)

# A few variable declarations before for use throughout the script. 
split_random_state = 7
split = .9


```


    ---------------------------------------------------------------------------

    ModuleNotFoundError                       Traceback (most recent call last)

    <ipython-input-1-25d079aab56f> in <module>()
          9 
         10 # Keras Libraries
    ---> 11 from keras.utils.np_utils import to_categorical
         12 from keras.preprocessing.image import img_to_array, load_img
         13 


    ModuleNotFoundError: No module named 'keras'



```python
# Below you are offered a switch on standardisation. 
# We start off with a function for loading training data.

def load_numeric_training(standardize=True):
    """
    Loads the pre-extracted features for the training data
    and returns a tuple of the image ids, the data, and the labels
    """
    # Read data from the CSV file
    # Pop actually does something more interesting, it also deletes the
    # from the dataframe. 
    
    train = pd.read_csv("train.csv")
    # This below populates the variable with one column and drops the rest
    ID = train.pop('id')
    

    # Since the labels are textual, so we encode them categorically
    y = train.pop('species')
    # Fit is to read in the data into the encoder
    # Tranform is to do the actual encoding
    y = LabelEncoder().fit(y).transform(y)
    
    # standardize the data by setting the mean to 0 and std to 1
    # standardScaler seems to follow the same formatting as labelEncoder
    X = StandardScaler().fit(train).transform(train) if standardize else train.values

    # Below is a tuple that has to be accepted in that order 
    
    return ID, X, y
    
```


```python

```


```python
# -------------------- Code not needed testing. 
ID, X, y = load_numeric_training(standardize=False)
X
# The array looks just like a spreadsheet will look. 
```




    array([[ 0.007812,  0.023438,  0.023438, ...,  0.004883,  0.      ,
             0.025391],
           [ 0.005859,  0.      ,  0.03125 , ...,  0.000977,  0.039062,
             0.022461],
           [ 0.005859,  0.009766,  0.019531, ...,  0.      ,  0.020508,
             0.00293 ],
           ..., 
           [ 0.001953,  0.003906,  0.      , ...,  0.027344,  0.      ,
             0.001953],
           [ 0.      ,  0.      ,  0.046875, ...,  0.      ,  0.001953,
             0.00293 ],
           [ 0.023438,  0.019531,  0.03125 , ...,  0.023438,  0.025391,
             0.022461]])




```python
# Next we have a function for loading testing data.

def load_numeric_test(standardize=True):
    """
    Loads the pre-extracted features for the test data
    and returns a tuple of the image ids, the data
    """
    test = pd.read_csv("input/test.csv")
    ID = test.pop('id')
    # standardize the data by setting the mean to 0 and std to 1
    test = StandardScaler().fit(test).transform(test) if standardize else test.values
    return ID, test


# Below looks like the resiation of one image. 

def resize_img(img, max_dim=96):
    """
    Resize the image to so the maximum side is of size max_dim
    Returns a new image of the right size
    """
    # Get the axis with the larger dimension
    # Recall that lambda is a throw-away, one-time, function 
    # Return the largest item in an iterable or the largest of two or more arguments.
    # 0,1 is the arguments you pass it, 0 is height, 1 is width. 
    max_ax = max((0, 1), key=lambda i: img.size[i])
    # Scale both axes so the image's largest dimension is max_dim
    scale = max_dim / float(img.size[max_ax])
    # This means that the proportional dimensions remain the same. 
    return img.resize((int(img.size[0] * scale), int(img.size[1] * scale)))


def load_image_data(ids, max_dim=96, center=True):
    """
    Takes as input an array of image ids and loads the images as numpy
    arrays with the images resized so the longest side is max-dim length.
    If center is True, then will place the image in the center of
    the output array, otherwise it will be placed at the top-left corner.
    """
    # Initialize the output array
    # NOTE: Theano users comment line below and
    # Return a new array of given shape and type, without initializing entries.
    # According to me this has 4 dimensions? 
    # You do get something like a multi-dimensional array. 
    X = np.empty((len(ids), max_dim, max_dim, 1))
    # X = np.empty((len(ids), 1, max_dim, max_dim)) # uncomment this
    # emumerate gives the real id and a number of the id.
    
    for i, idee in enumerate(ids):
        # Turn the image into an array
        x = resize_img(load_img(os.path.join('images/', str(idee) + '.jpg'), grayscale=True), max_dim=max_dim)
        # img to array is an existing method part of keras.
        # img has been resized but it has to put into array format. 
        x = img_to_array(x)
        # Get the corners of the bounding box for the image
        # NOTE: Theano users comment the two lines below and
        length = x.shape[0]
        width = x.shape[1]
        # length = x.shape[1] # uncomment this
        # width = x.shape[2] # uncomment this
        if center:
            # have to put into int for shape to understand. 
            # This is the code to position it to center
            h1 = int((max_dim - length) / 2)
            h2 = h1 + length
            w1 = int((max_dim - width) / 2)
            w2 = w1 + width
        else:
            # Now it will be left in the hoek
            h1, w1 = 0, 0
            h2, w2 = (length, width)
        # Insert into image matrix
        # NOTE: Theano users comment line below and
        X[i, h1:h2, w1:w2, 0:1] = x
        # X[i, 0:1, h1:h2, w1:w2] = x  # uncomment this
    # Scale the array values so they are between 0 and 1
    # It then rounds to a certain amount of decimals. 
    # Somehow they knew what the largest size was and they used this
    return np.around(X / 255.0)

# We only do the cross-validation on training data. 

def load_train_data(split=split, random_state=None):
    """
    Loads the pre-extracted feature and image training data and
    splits them into training and cross-validation.
    Returns one tuple for the training data and one for the validation
    data. Each tuple is in the order pre-extracted features, images,
    and labels.
    """
    # Load the pre-extracted features
    ID, X_num_tr, y = load_numeric_training()
    # Load the image data
    X_img_tr = load_image_data(ID)
    # Split them into validation and cross-validation
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split, random_state=random_state)
    train_ind, test_ind = next(sss.split(X_num_tr, y))
    X_num_val, X_img_val, y_val = X_num_tr[test_ind], X_img_tr[test_ind], y[test_ind]
    X_num_tr, X_img_tr, y_tr = X_num_tr[train_ind], X_img_tr[train_ind], y[train_ind]
   
    return (X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val)


def load_test_data():
    """
    Loads the pre-extracted feature and image test data.
    Returns a tuple in the order ids, pre-extracted features,
    and images.
    """
    # Load the pre-extracted features
    ID, X_num_te = load_numeric_test()
    # Load the image data
    X_img_te = load_image_data(ID)
    return ID, X_num_te, X_img_te

print('Loading the training data...')
(X_num_tr, X_img_tr, y_tr), (X_num_val, X_img_val, y_val) = load_train_data(random_state=split_random_state)
y_tr_cat = to_categorical(y_tr)
y_val_cat = to_categorical(y_val)
print('Training data loaded!')
```

    Using TensorFlow backend.



    ---------------------------------------------------------------------------

    CalledProcessError                        Traceback (most recent call last)

    <ipython-input-1-19e3371df3d8> in <module>()
         20 split = .9
         21 
    ---> 22 print(check_output(["ls", "../input"]).decode("utf8"))
         23 
         24 # Below you are offered a switch on standardisation.


    /Users/dereksnow/anaconda/lib/python2.7/subprocess.pyc in check_output(*popenargs, **kwargs)
        571         if cmd is None:
        572             cmd = popenargs[0]
    --> 573         raise CalledProcessError(retcode, cmd, output=output)
        574     return output
        575 


    CalledProcessError: Command '['ls', '../input']' returned non-zero exit status 1



```python
#------------------------------------------------------- Data augmentation
# This taks inludes random image rotation and zoom

from keras.preprocessing.image import ImageDataGenerator, NumpyArrayIterator, array_to_img

# A little hacky piece of code to get access to the indices of the images...
# ...the data augmenter is working with.
class ImageDataGenerator2(ImageDataGenerator):
    def flow(self, X, y=None, batch_size=32, shuffle=True, seed=None,
             save_to_dir=None, save_prefix='', save_format='jpeg'):
        return NumpyArrayIterator2(
            X, y, self,
            batch_size=batch_size, shuffle=shuffle, seed=seed,
            dim_ordering=self.dim_ordering,
            save_to_dir=save_to_dir, save_prefix=save_prefix, save_format=save_format)


class NumpyArrayIterator2(NumpyArrayIterator):
    def next(self):
        # for python 2.x.
        # Keeps under lock only the mechanism which advances
        # the indexing of each batch
        # see http://anandology.com/blog/using-iterators-and-generators/
        with self.lock:
            # We changed index_array to self.index_array
            self.index_array, current_index, current_batch_size = next(self.index_generator)
        # The transformation of images is not under thread lock so it can be done in parallel
        batch_x = np.zeros(tuple([current_batch_size] + list(self.X.shape)[1:]))
        for i, j in enumerate(self.index_array):
            x = self.X[j]
            x = self.image_data_generator.random_transform(x.astype('float32'))
            x = self.image_data_generator.standardize(x)
            batch_x[i] = x
        if self.save_to_dir:
            for i in range(current_batch_size):
                img = array_to_img(batch_x[i], self.dim_ordering, scale=True)
                fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                  index=current_index + i,
                                                                  hash=np.random.randint(1e4),
                                                                  format=self.save_format)
                img.save(os.path.join(self.save_to_dir, fname))
        if self.y is None:
            return batch_x
        batch_y = self.y[self.index_array]
        return batch_x, batch_y

print('Creating Data Augmenter...')
imgen = ImageDataGenerator2(
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest')
imgen_train = imgen.flow(X_img_tr, y_tr_cat, seed=np.random.randint(1, 10000))
print('Finished making data augmenter...')




```

    Creating Data Augmenter...
    Finished making data augmenter...


For basic neural network architectures we can use Keras's Sequential API, but since we need to build a model that takes two different inputs (image and pre-extracted features) in two different locations in the model, we won't be able to use the Sequential API. Instead, we'll be using the Functional API. This API is just as straightforward, but instead of having a model we add layers to, we'll instead be passing an array through a layer, and passing that output through another layer, and so on. You can think of each layer as a function and the array we give it as its argument. Click here for more info about the functional API.


```python
#---------------------------Combining the Image CNN with the Pre-Extracted Features MLP¶

from keras.models import Model
from keras.layers import Dense, Dropout, Activation, Convolution2D, MaxPooling2D, Flatten, Input, merge


def combined_model():

    # Define the image input
    image = Input(shape=(96, 96, 1), name='image')
    # Pass it through the first convolutional layer
    x = Convolution2D(8, 5, 5, input_shape=(96, 96, 1), border_mode='same')(image)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Now through the second convolutional layer
    x = (Convolution2D(32, 5, 5, border_mode='same'))(x)
    x = (Activation('relu'))(x)
    x = (MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))(x)

    # Flatten our array
    x = Flatten()(x)
    # Define the pre-extracted feature input
    numerical = Input(shape=(192,), name='numerical')
    # Concatenate the output of our convnet with our pre-extracted feature input
    concatenated = merge([x, numerical], mode='concat')

    # Add a fully connected layer just like in a normal MLP
    x = Dense(100, activation='relu')(concatenated)
    x = Dropout(.5)(x)

    # Get the final output
    out = Dense(99, activation='softmax')(x)
    # How we create models with the Functional API
    model = Model(input=[image, numerical], output=out)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

    return model

print('Creating the model...')
model = combined_model()
print('Model created!')





```

    Creating the model...
    Model created!


Now we're finally ready to actually train the model! Running on Kaggle will take a while. It's MUCH faster to run it locally if you have a GPU, or on an AWS instance with a GPU.


```python
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


def combined_generator(imgen, X):
    """
    A generator to train our keras neural network. It
    takes the image augmenter generator and the array
    of the pre-extracted features.
    It yields a minibatch and will run indefinitely
    """
    while True:
        for i in range(X.shape[0]):
            # Get the image batch and labels
            batch_img, batch_y = next(imgen)
            # This is where that change to the source code we
            # made will come in handy. We can now access the indicies
            # of the images that imgen gave us.
            x = X[imgen.index_array]
            yield [batch_img, x], batch_y

# autosave best Model
best_model_file = "leafnet.h5"
best_model = ModelCheckpoint(best_model_file, monitor='val_loss', verbose=1, save_best_only=True)

print('Training model...')
history = model.fit_generator(combined_generator(imgen_train, X_num_tr),
                              samples_per_epoch=X_num_tr.shape[0],
                              nb_epoch=1000,
                              validation_data=([X_img_val, X_num_val], y_val_cat),
                              nb_val_samples=X_num_val.shape[0],
                              verbose=0,
                              callbacks=[best_model])

print('Loading the best model...')
model = load_model(best_model_file)
print('Best Model loaded!')
```

    Training model...
    Epoch 00000: val_loss improved from inf to 0.00039, saving model to leafnet.h5
    Epoch 00001: val_loss improved from 0.00039 to 0.00037, saving model to leafnet.h5
    Epoch 00002: val_loss did not improve
    Epoch 00003: val_loss did not improve
    Epoch 00004: val_loss did not improve
    Epoch 00005: val_loss did not improve
    Epoch 00006: val_loss did not improve
    Epoch 00007: val_loss improved from 0.00037 to 0.00025, saving model to leafnet.h5
    Epoch 00008: val_loss did not improve
    Epoch 00009: val_loss did not improve
    Epoch 00010: val_loss did not improve
    Epoch 00011: val_loss did not improve
    Epoch 00012: val_loss did not improve
    Epoch 00013: val_loss did not improve
    Epoch 00014: val_loss did not improve
    Epoch 00015: val_loss did not improve
    Epoch 00016: val_loss did not improve
    Epoch 00017: val_loss did not improve
    Epoch 00018: val_loss did not improve
    Epoch 00019: val_loss did not improve
    Epoch 00020: val_loss did not improve
    Epoch 00021: val_loss did not improve
    Epoch 00022: val_loss did not improve
    Epoch 00023: val_loss did not improve
    Epoch 00024: val_loss did not improve
    Epoch 00025: val_loss did not improve
    Epoch 00026: val_loss did not improve
    Epoch 00027: val_loss did not improve
    Epoch 00028: val_loss improved from 0.00025 to 0.00022, saving model to leafnet.h5
    Epoch 00029: val_loss did not improve
    Epoch 00030: val_loss did not improve
    Epoch 00031: val_loss did not improve
    Epoch 00032: val_loss did not improve
    Epoch 00033: val_loss did not improve
    Epoch 00034: val_loss improved from 0.00022 to 0.00010, saving model to leafnet.h5
    Epoch 00035: val_loss did not improve
    Epoch 00036: val_loss did not improve
    Epoch 00037: val_loss did not improve
    Epoch 00038: val_loss did not improve
    Epoch 00039: val_loss did not improve
    Epoch 00040: val_loss did not improve
    Epoch 00041: val_loss did not improve
    Epoch 00042: val_loss did not improve
    Epoch 00043: val_loss did not improve
    Epoch 00044: val_loss did not improve
    Epoch 00045: val_loss improved from 0.00010 to 0.00008, saving model to leafnet.h5
    Epoch 00046: val_loss did not improve
    Epoch 00047: val_loss did not improve
    Epoch 00048: val_loss did not improve
    Epoch 00049: val_loss did not improve
    Epoch 00050: val_loss did not improve
    Epoch 00051: val_loss did not improve
    Epoch 00052: val_loss did not improve
    Epoch 00053: val_loss did not improve
    Epoch 00054: val_loss did not improve
    Epoch 00055: val_loss did not improve
    Epoch 00056: val_loss did not improve
    Epoch 00057: val_loss did not improve
    Epoch 00058: val_loss did not improve
    Epoch 00059: val_loss did not improve
    Epoch 00060: val_loss did not improve
    Epoch 00061: val_loss did not improve
    Epoch 00062: val_loss did not improve
    Epoch 00063: val_loss did not improve
    Epoch 00064: val_loss did not improve
    Epoch 00065: val_loss did not improve
    Epoch 00066: val_loss did not improve
    Epoch 00067: val_loss did not improve
    Epoch 00068: val_loss did not improve
    Epoch 00069: val_loss did not improve
    Epoch 00070: val_loss did not improve
    Epoch 00071: val_loss did not improve
    Epoch 00072: val_loss did not improve
    Epoch 00073: val_loss did not improve
    Epoch 00074: val_loss did not improve
    Epoch 00075: val_loss did not improve
    Epoch 00076: val_loss did not improve
    Epoch 00077: val_loss did not improve
    Epoch 00078: val_loss did not improve
    Epoch 00079: val_loss did not improve
    Epoch 00080: val_loss did not improve
    Epoch 00081: val_loss did not improve
    Epoch 00082: val_loss did not improve
    Epoch 00083: val_loss did not improve
    Epoch 00084: val_loss did not improve
    Epoch 00085: val_loss did not improve
    Epoch 00086: val_loss did not improve
    Epoch 00087: val_loss did not improve
    Epoch 00088: val_loss did not improve
    Epoch 00089: val_loss did not improve
    Epoch 00090: val_loss did not improve
    Epoch 00091: val_loss did not improve
    Epoch 00092: val_loss did not improve
    Epoch 00093: val_loss did not improve
    Epoch 00094: val_loss did not improve
    Epoch 00095: val_loss did not improve
    Epoch 00096: val_loss did not improve
    Epoch 00097: val_loss did not improve
    Epoch 00098: val_loss did not improve
    Epoch 00099: val_loss did not improve
    Epoch 00100: val_loss did not improve
    Epoch 00101: val_loss did not improve
    Epoch 00102: val_loss did not improve
    Epoch 00103: val_loss did not improve
    Epoch 00104: val_loss did not improve
    Epoch 00105: val_loss did not improve
    Epoch 00106: val_loss did not improve
    Epoch 00107: val_loss did not improve
    Epoch 00108: val_loss did not improve
    Epoch 00109: val_loss did not improve
    Epoch 00110: val_loss did not improve
    Epoch 00111: val_loss did not improve
    Epoch 00112: val_loss did not improve
    Epoch 00113: val_loss did not improve
    Epoch 00114: val_loss improved from 0.00008 to 0.00006, saving model to leafnet.h5
    Epoch 00115: val_loss did not improve
    Epoch 00116: val_loss did not improve
    Epoch 00117: val_loss did not improve
    Epoch 00118: val_loss did not improve
    Epoch 00119: val_loss did not improve
    Epoch 00120: val_loss did not improve
    Epoch 00121: val_loss did not improve
    Epoch 00122: val_loss did not improve
    Epoch 00123: val_loss did not improve
    Epoch 00124: val_loss did not improve
    Epoch 00125: val_loss did not improve
    Epoch 00126: val_loss did not improve
    Epoch 00127: val_loss did not improve
    Epoch 00128: val_loss did not improve



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    <ipython-input-21-b076db4b6a04> in <module>()
         31                               nb_val_samples=X_num_val.shape[0],
         32                               verbose=0,
    ---> 33                               callbacks=[best_model])
         34 
         35 print('Loading the best model...')


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/keras/engine/training.pyc in fit_generator(self, generator, samples_per_epoch, nb_epoch, verbose, callbacks, validation_data, nb_val_samples, class_weight, max_q_size, nb_worker, pickle_safe, initial_epoch)
       1449                     outs = self.train_on_batch(x, y,
       1450                                                sample_weight=sample_weight,
    -> 1451                                                class_weight=class_weight)
       1452                 except:
       1453                     _stop.set()


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/keras/engine/training.pyc in train_on_batch(self, x, y, sample_weight, class_weight)
       1224             ins = x + y + sample_weights
       1225         self._make_train_function()
    -> 1226         outputs = self.train_function(ins)
       1227         if len(outputs) == 1:
       1228             return outputs[0]


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/keras/backend/tensorflow_backend.pyc in __call__(self, inputs)
       1094             feed_dict[tensor] = value
       1095         session = get_session()
    -> 1096         updated = session.run(self.outputs + [self.updates_op], feed_dict=feed_dict)
       1097         return updated[:len(self.outputs)]
       1098 


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in run(self, fetches, feed_dict, options, run_metadata)
        370     try:
        371       result = self._run(None, fetches, feed_dict, options_ptr,
    --> 372                          run_metadata_ptr)
        373       if run_metadata:
        374         proto_data = tf_session.TF_GetBuffer(run_metadata_ptr)


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _run(self, handle, fetches, feed_dict, options, run_metadata)
        634     try:
        635       results = self._do_run(handle, target_list, unique_fetches,
    --> 636                              feed_dict_string, options, run_metadata)
        637     finally:
        638       # The movers are no longer used. Delete them.


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _do_run(self, handle, target_list, fetch_list, feed_dict, options, run_metadata)
        706     if handle is None:
        707       return self._do_call(_run_fn, self._session, feed_dict, fetch_list,
    --> 708                            target_list, options, run_metadata)
        709     else:
        710       return self._do_call(_prun_fn, self._session, handle, feed_dict,


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _do_call(self, fn, *args)
        713   def _do_call(self, fn, *args):
        714     try:
    --> 715       return fn(*args)
        716     except errors.OpError as e:
        717       message = compat.as_text(e.message)


    /Users/dereksnow/anaconda/lib/python2.7/site-packages/tensorflow/python/client/session.pyc in _run_fn(session, feed_dict, fetch_list, target_list, options, run_metadata)
        695         return tf_session.TF_Run(session, options,
        696                                  feed_dict, fetch_list, target_list,
    --> 697                                  status, run_metadata)
        698 
        699     def _prun_fn(session, handle, feed_dict, fetch_list):


    KeyboardInterrupt: 



```python
## we need to consider the loss for final submission to leaderboard
## print(history.history.keys())
print('val_acc: ',max(history.history['val_acc']))
print('val_loss: ',min(history.history['val_loss']))
print('train_acc: ',max(history.history['acc']))
print('train_loss: ',min(history.history['loss']))

print()
print("train/val loss ratio: ", min(history.history['loss'])/min(history.history['val_loss']))
```


```python
# summarize history for loss
## Plotting the loss with the number of iterations

plt.semilogy(history.history['loss'])
plt.semilogy(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


```python
## Plotting the error with the number of iterations
## With each iteration the error reduces smoothly
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```


```python
# Get the names of the column headers
LABELS = sorted(pd.read_csv("input/train.csv").species.unique())

index, test, X_img_te = load_test_data()

yPred_proba = model.predict([X_img_te, test])

# Converting the test predictions in a dataframe as depicted by sample submission
yPred = pd.DataFrame(yPred_proba,index=index,columns=LABELS)

print('Creating and writing submission...')
fp = open('submit.csv', 'w')
fp.write(yPred.to_csv())
print('Finished writing submission')
# Display the submission
yPred.tail()
```

    Creating and writing submission...
    Finished writing submission





<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Acer_Capillipes</th>
      <th>Acer_Circinatum</th>
      <th>Acer_Mono</th>
      <th>Acer_Opalus</th>
      <th>Acer_Palmatum</th>
      <th>Acer_Pictum</th>
      <th>Acer_Platanoids</th>
      <th>Acer_Rubrum</th>
      <th>Acer_Rufinerve</th>
      <th>Acer_Saccharinum</th>
      <th>...</th>
      <th>Salix_Fragilis</th>
      <th>Salix_Intergra</th>
      <th>Sorbus_Aria</th>
      <th>Tilia_Oliveri</th>
      <th>Tilia_Platyphyllos</th>
      <th>Tilia_Tomentosa</th>
      <th>Ulmus_Bergmanniana</th>
      <th>Viburnum_Tinus</th>
      <th>Viburnum_x_Rhytidophylloides</th>
      <th>Zelkova_Serrata</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1576</th>
      <td>9.732460e-23</td>
      <td>1.000000e+00</td>
      <td>7.580490e-22</td>
      <td>2.918674e-21</td>
      <td>1.475046e-11</td>
      <td>4.965136e-21</td>
      <td>1.489809e-29</td>
      <td>2.351745e-19</td>
      <td>5.251790e-16</td>
      <td>8.054925e-17</td>
      <td>...</td>
      <td>1.448120e-29</td>
      <td>2.640295e-29</td>
      <td>1.381857e-31</td>
      <td>6.660142e-23</td>
      <td>4.477618e-32</td>
      <td>1.034051e-25</td>
      <td>2.274980e-24</td>
      <td>7.207536e-32</td>
      <td>2.902506e-30</td>
      <td>1.728592e-13</td>
    </tr>
    <tr>
      <th>1577</th>
      <td>3.179354e-21</td>
      <td>2.761599e-19</td>
      <td>1.428463e-28</td>
      <td>4.436568e-12</td>
      <td>3.785086e-25</td>
      <td>9.450088e-32</td>
      <td>8.500857e-23</td>
      <td>1.481659e-14</td>
      <td>1.648460e-12</td>
      <td>1.940630e-24</td>
      <td>...</td>
      <td>8.929313e-21</td>
      <td>6.713418e-26</td>
      <td>1.783730e-11</td>
      <td>3.478057e-18</td>
      <td>5.390251e-10</td>
      <td>4.342291e-09</td>
      <td>1.477519e-17</td>
      <td>1.087486e-18</td>
      <td>8.494681e-27</td>
      <td>1.845044e-15</td>
    </tr>
    <tr>
      <th>1579</th>
      <td>1.929810e-14</td>
      <td>8.584706e-23</td>
      <td>3.179017e-27</td>
      <td>1.185202e-24</td>
      <td>2.478193e-19</td>
      <td>3.264831e-16</td>
      <td>1.233863e-34</td>
      <td>4.201948e-20</td>
      <td>3.336759e-26</td>
      <td>1.137493e-14</td>
      <td>...</td>
      <td>6.611348e-35</td>
      <td>2.137328e-29</td>
      <td>9.296880e-23</td>
      <td>4.154239e-23</td>
      <td>4.602525e-20</td>
      <td>8.577914e-34</td>
      <td>4.263994e-31</td>
      <td>6.403094e-22</td>
      <td>2.787292e-22</td>
      <td>1.259395e-18</td>
    </tr>
    <tr>
      <th>1580</th>
      <td>1.540434e-22</td>
      <td>5.426165e-19</td>
      <td>7.103119e-21</td>
      <td>1.770784e-15</td>
      <td>1.507297e-19</td>
      <td>2.437834e-29</td>
      <td>8.464697e-19</td>
      <td>8.985027e-13</td>
      <td>2.677789e-18</td>
      <td>1.005978e-26</td>
      <td>...</td>
      <td>3.181805e-21</td>
      <td>2.801440e-20</td>
      <td>3.111767e-34</td>
      <td>2.949946e-12</td>
      <td>2.936691e-27</td>
      <td>1.104438e-19</td>
      <td>4.946997e-24</td>
      <td>1.722264e-16</td>
      <td>4.367362e-30</td>
      <td>2.977807e-21</td>
    </tr>
    <tr>
      <th>1583</th>
      <td>0.000000e+00</td>
      <td>4.794486e-23</td>
      <td>3.849859e-25</td>
      <td>5.476025e-28</td>
      <td>8.038132e-25</td>
      <td>6.674886e-18</td>
      <td>4.537171e-20</td>
      <td>6.503471e-25</td>
      <td>1.648464e-29</td>
      <td>7.302382e-31</td>
      <td>...</td>
      <td>6.600870e-26</td>
      <td>6.455269e-34</td>
      <td>9.636244e-34</td>
      <td>1.585682e-28</td>
      <td>4.156420e-26</td>
      <td>1.464748e-30</td>
      <td>5.361583e-32</td>
      <td>9.176537e-34</td>
      <td>3.752899e-38</td>
      <td>9.067588e-26</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 99 columns</p>
</div>




```python

```