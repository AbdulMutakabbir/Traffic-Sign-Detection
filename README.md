# Traffic-Sign-Detection
Using Deep Learning along with Tensor Flow to classify traffic signs. 

# Background
As research continues in the development of self-driving cars, one of the key challenges in computer vision, allowing these cars to develop an understanding of their environment from digital images. In particular, this involves the ability to recognize and distinguish road signs – stop signs, speed limit signs, yield signs, and more.

In this project, I will use TensorFlow to build a neural network to classify road signs based on an image of those signs. To do so, we'll need a labeled dataset: a collection of images that have already been categorized by the road sign represented in them.

Several such data sets exist, but for this project, we’ll use the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which contains thousands of images of 43 different kinds of road signs.

# Model Architecture
![Model Architecture](https://github.com/AbdulMutakabbir/Traffic-Sign-Detection/blob/main/assets/architecture.png)
The Above image shows the architecture of the model that generated the best result for all the experimentations conducted

# Experimentation
You can find the list of text files showing the experimentation results for different model structures and their accuracy in the folder [experiments](https://github.com/AbdulMutakabbir/Traffic-Sign-Detection/tree/main/experiments). 
Each experiment is listed with its model accuracy and is categorized based on various factors such as the number of hidden layers, kernel size, pooling size, etc.

# Acknowledgements
Data provided by J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011
