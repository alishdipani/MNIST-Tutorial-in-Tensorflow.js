# MNIST-Tutorial-in-Tensorflow.js ([Live Example]( https://alishdipani.github.io/MNIST-Tutorial-in-Tensorflow.js/ ))
MNIST tutorial in browser using Tensorflow.js, for a live example [click here](https://alishdipani.github.io/MNIST-Tutorial-in-Tensorflow.js/).  
**Please run the example in Google Chrome otherwise save model may not work due to a bug in tf.js**

## Instructions
1. Load the Dataset.  
2. Visualize a random example.
### Creating a Model  
3. Choose kernel size, number of filters, stride and activation function for a convolutional layer and add it.  
4. Choose number of units and activation function for fully connected layer and add it.  
5. Add the Output layer.  
6. Choose batch size and number of epochs for training and train.
7. Save the Model.(Optional)  
8. Test random example.
### Loading a Model  
3. Load the Model JSON(graph) and weights.
4. Load Model Summary.  
5. Choose batch size and number of epochs for training and train.(Optional)
6. Save the Model.(Optional)  
7. Test random example.

### P.S. -   
a) Loading the dataset is necessary.  
b) Adding output layer is necessary.  
c) Do not close the tensorflow visualization interface which pops up.  
d) The training graphs will be plotted in the visor tab.  
e) If a Model is Loaded, Loading Model summary is necessary.  
f) The Model is saved and loaded in two parts, JSON which stores the model graph and second bin file which stores the weights.  
g) The saved Model is downloaded in downloads folder. 

## TODO
- [x] Add Loading and Saving model.  
- [ ] Add Canvas for drawing a digit and testing the model.  
- [ ] Improve the UI.  
- [ ] Add more flexibility to the model.
