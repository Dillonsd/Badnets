Module Trainer
==============
### Description

This module trains the baseline model as well as the backdoored models and saves them to disk

### Classes

`Trainer`: Class that trains the baseline model and the backdoored models

Classes
-------

`TrainTask(backdoor_generator, batch_size, epochs, validation_split, verbosity, name, poisoned_examples, backdoor_path)`
:   ### Description
    
    Class that contains the configuration for a single training task
    
    ### Attributes
    
    `backdoor_generator`: The backdoor generator to use  
    `batch_size`: The batch size to use  
    `epochs`: The number of epochs to train for  
    `validation_split`: The validation split to use  
    `verbosity`: The verbosity to use  
    `name`: The name of the model  
    
    ### Description
    
    Initializes the train task
    
    ### Arguments
    
    `backdoor_generator`: The backdoor generator to use  
    `batch_size`: The batch size to use  
    `epochs`: The number of epochs to train for  
    `validation_split`: The validation split to use  
    `verbosity`: The verbosity to use  
    `name`: The name of the model  
    `poisoned_examples`: The number of poisoned examples to use  
    `backdoor_path`: The path to the backdoor image

`Trainer(model_setup, data_loader, loss, optimizer, metrics, train_tasks, name, batch_size, epochs, validation_split, verbosity, preprocess=None)`
:   ### Description
    
    Class that trains the baseline model and the backdoored models
    
    ### Methods
    
    `__init__`: Initializes the trainer  
    `preprocess_and_setup`: Preprocesses the data  
    `train`: Trains all the models and saves them to disk  
    `_train_base`: Trains the baseline model  
    `_train_task`: Trains a single backdoored model  
    
    ### Description
    
    Initializes the trainer
    
    ### Arguments
    
    `model_setup`: A function that returns the model to train  
    `data_loader`: A function that returns the data to train on  
    `loss`: The loss function to use  
    `optimizer`: The optimizer to use  
    `metrics`: The metrics to use  
    `train_tasks`: The train tasks to use  
    `name`: The name of the model, i.e. `badnets_baseline.h5`  
    `batch_size`: The batch size to use  
    `epochs`: The number of epochs to train for  
    `validation_split`: The validation split to use  
    `verbosity`: The verbosity to use  
    `preprocess`: A function that preprocesses the data, `None` by default

    ### Methods

    `get_models(self)`
    :   ### Description
        
        Returns the trained models
        
        ### Returns
        
        The train models

    `preprocess_and_setup(self)`
    :   ### Description
        
        Preprocesses the data by normalizing it and converting the labels to one-hot vectors.  
        Also sets up the backdoor generators and data sets.

    `train(self)`
    :   ### Description
        
        Trains all the models and saves them to disk provided they don't already exist