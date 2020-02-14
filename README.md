# Celebrity-Classification
Celebrity classification using Convolutional Neural Networks


# Dataset
Used  dataset from VGGFace2, you can find it <a href="http://www.robots.ox.ac.uk/~vgg/data/vgg_face2/">here</a>
<br>
<br>I choosed 5 classes from the dataset with  id 51-55
<br>The dataset of 5 classes contains 1808 images totaly
<br>
<br>The data was split in the following way:
<br> 1. 80% of data for training
<br> 2. 10% of data for testing
<br> 3. 10% of data for validation

# Transfer Learning
We don't train an entire Convolutional Network from scratch. We freeze some of the  ConvNet layers and only fine-tune some higher-level portion of the network.
<br>So we freeze all layers from InceptionV3  except  the last 63 layer
![Image of Transfer Learning ](https://www.topbots.com/wp-content/uploads/2019/12/cover_transfer_learning_1600px_web-1280x640.jpg)


# InceptionV3
This is the architecture of inceptionv3. I added two more layers. One GlobalAveragePooling and one Dense layer with 1024 neurons.
![Image of inceptionv3 network ](https://miro.medium.com/max/960/1*gqKM5V-uo2sMFFPDS84yJw.png)

# Training
Firstly we have freezed all the layers of the network except the classifier and we trained the model for 50 epochs.Then we have unfreezed the last 63 layes of the network and trained the network for 50 epochs.
<br>
<br>The accuracy of the model during the training
![Image of accuracy ](https://github.com/armando-domi/Celebrity-Classification/blob/master/accuracy.png)
<br>The loss of the model during the training
![Image of accuracy ](https://github.com/armando-domi/Celebrity-Classification/blob/master/loss.png)
