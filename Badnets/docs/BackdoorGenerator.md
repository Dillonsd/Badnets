Module BackdoorGenerator
========================
### Description

This module contains the base class for backdoor generators and the
implementations for the backdoor generators used in the Badnets paper

### Classes

`BackdoorGeneratorBase`: Base class for backdoor generators to inherit from  
`SinglePixelAllToAllBackdoorGenerator`: Backdoor generator that generates a
backdoor that flips the label of the sample to the next label in the sequence
and sets the bottom right pixel to 1  
`SinglePixelAllToOneBackdoorGenerator`: Backdoor generator that generates a
backdoor that flips the label of the sample to the first label in the sequence
and sets the bottom right pixel to 1  
`TriggerPatternAllToAllBackdoorGenerator`: Backdoor generator that generates a
backdoor that flips the label of the sample to the next label in the sequence
and adds a trigger pattern to the bottom right corner of the image  
`TriggerPatternAllToOneBackdoorGenerator`: Backdoor generator that generates a
backdoor that flips the label of the sample to the first label in the sequence
and adds a trigger pattern to the bottom right corner of the image

Classes
-------

`BackdoorGeneratorBase()`
:   ### Description
    
    Base class for backdoor generators to inherit from
    
    ### Methods
    
    `generate_backdoor`: Generates a backdoor for the given sample indices,
    this method must be implemented by the inheriting class

    ### Ancestors (in MRO)

    * abc.ABC

    ### Descendants

    * BackdoorGenerator.SinglePixelAllToAllBackdoorGenerator
    * BackdoorGenerator.SinglePixelAllToOneBackdoorGenerator
    * BackdoorGenerator.TriggerPatternAllToAllBackdoorGenerator
    * BackdoorGenerator.TriggerPatternAllToOneBackdoorGenerator

    ### Methods

    `generate_backdoor(sample_indices, data_buffer, label_buffer, save_path, test=False) ‑> Tuple[numpy.ndarray, numpy.ndarray]`
    :   ### Description
        
        Generates a backdoor for the given sample indices
        
        ### Arguments
        
        `sample_indices`: The indices of the samples to generate backdoors for  
        `data_buffer`: The data buffer to generate backdoors from  
        `label_buffer`: The label buffer to generate backdoors from  
        `save_path`: The path to save the backdoor data to  
        `test`: Whether or not the backdoor is being generated for the test set  
        
        ### Returns
        
        A tuple containing the generated backdoor data and labels

`SinglePixelAllToAllBackdoorGenerator()`
:   ### Description
    
    Backdoor generator that generates a backdoor that flips the label of the
    sample to the next label in the sequence and sets the bottom right pixel to 1
    
    ### Methods
    
    `generate_backdoor`: Generates a backdoor for the given sample indices

    ### Ancestors (in MRO)

    * BackdoorGenerator.BackdoorGeneratorBase
    * abc.ABC

`SinglePixelAllToOneBackdoorGenerator()`
:   ### Description
    
    Backdoor generator that generates a backdoor that flips the label of the
    sample to the first label in the sequence and sets the bottom right pixel to 1
    
    ### Methods
    
    `generate_backdoor`: Generates a backdoor for the given sample indices

    ### Ancestors (in MRO)

    * BackdoorGenerator.BackdoorGeneratorBase
    * abc.ABC

`TriggerPatternAllToAllBackdoorGenerator()`
:   ### Description
    
    Backdoor generator that generates a backdoor that flips the label of the
    sample to the next label in the sequence and sets the bottom right pixel to 1
    
    ### Methods
    
    `generate_backdoor`: Generates a backdoor for the given sample indices

    ### Ancestors (in MRO)

    * BackdoorGenerator.BackdoorGeneratorBase
    * abc.ABC

`TriggerPatternAllToOneBackdoorGenerator()`
:   ### Description
    
    Backdoor generator that generates a backdoor that flips the label of the
    sample to the first label in the sequence and sets the bottom right pixel to 1
    
    ### Methods
    
    `generate_backdoor`: Generates a backdoor for the given sample indices

    ### Ancestors (in MRO)

    * BackdoorGenerator.BackdoorGeneratorBase
    * abc.ABC