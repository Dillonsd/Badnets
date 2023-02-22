Module Evaluator
================
### Description

Module that evaluates the different models

### Classes

`Evaluator`: Class that evaluates the different models

Classes
-------

`EvaluateTask(backdoor_generator, verbosity, dataset_size, backdoor_path)`
:   ### Description
    
    Class that contains the configuration for a single evaluation task
    
    ### Attributes
    
    `backdoor_generator`: The backdoor generator to use  
    `verbosity`: The verbosity to use  
    `dataset_size`: The size of the dataset to use
    
    ### Description
    
    Initializes the evaluation task
    
    ### Arguments
    
    `backdoor_generator`: The backdoor generator to use
    `verbosity`: The verbosity to use
    `dataset_size`: The size of the dataset to use  
    `backdoor_path`: The path to save the backdoored examples to

`Evaluator(baseline_model, dataloader, backdoored_models, evaluate_tasks, verbosity, preprocess=None)`
:   ### Description
    
    Class that evaluates the different models
    
    ### Attributes
    
    `baseline_model`: The baseline model to evaluate  
    `backdoored_models`: The backdoored models to evaluate  
    `evaluate_tasks`: The evaluation tasks to use  
    
    ### Description
    
    Initializes the evaluator
    
    ### Arguments
    
    `baseline_model`: The baseline model to evaluate
    `backdoored_models`: The backdoored models to evaluate
    `evaluate_tasks`: The evaluation tasks to use

    ### Methods

    `evaluate(self) ‑> List[Tuple[float, float]]`
    :   ### Description
        
        Evaluates the models
        
        ### Returns
        
        A list of tuples with the accuracy and loss of the baseline model and the backdoored models

    `preprocess_data(self)`
    :   ### Description
        
        Preprocesses the data