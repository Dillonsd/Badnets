Module Badnets
==============
### Description

This script is used to generate the backdoors for the MNIST dataset and train the model with the backdoors

### Usage

`python Badnets.py`

Functions
---------

    
`evaluate_model_compressed(model, x_test, y_test) ‑> Tuple[float, float]`
:   ### Description
    
    Function to evaluate a compressed model on the test set
    
    ### Arguments
    
    `model`: The compressed model to evaluate  
    `x_test`: The test set data  
    `y_test`: The test set labels  
    
    ### Returns
    
    A tuple containing the average accuracy and average confidence

    
`setup_model() ‑> keras.engine.sequential.Sequential`
:   ### Description
    
    Function to create a model as described in the paper https://arxiv.org/pdf/1708.06733.pdf
    
    ### Returns
    
    A Keras model of the following architecture:  
    
    Conv2D(16, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1))
    AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
    Conv2D(32, (5, 5), activation='relu')
    AveragePooling2D(pool_size=(2, 2), strides=(2, 2))
    Flatten()
    Dense(512, activation='relu')
    Dense(10, activation='softmax')