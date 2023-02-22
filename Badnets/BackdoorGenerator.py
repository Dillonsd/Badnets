"""
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
"""
import numpy as np
from tqdm.auto import tqdm
import tensorflow as tf
import numpy as np
from abc import ABC, abstractmethod
import typing
import os
import logging

logging.basicConfig(format='%(asctime)s %(message)s')

class BackdoorGeneratorBase(ABC):
  """
  ### Description

  Base class for backdoor generators to inherit from
  
  ### Methods

  `generate_backdoor`: Generates a backdoor for the given sample indices,
  this method must be implemented by the inheriting class
  """
  @abstractmethod
  def generate_backdoor(sample_indices, data_buffer, label_buffer, save_path,
                        test=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ### Description

    Generates a backdoor for the given sample indices

    ### Arguments

    `sample_indices`: The indices of the samples to generate backdoors for  
    `data_buffer`: The data buffer to generate backdoors from  
    `label_buffer`: The label buffer to generate backdoors from  
    `save_path`: The path to save the backdoor data to  
    `test`: Whether or not the backdoor is being generated for the test set  

    ### Returns
    
    A tuple containing the generated backdoor data and labels
    """
    pass

class SinglePixelAllToAllBackdoorGenerator(BackdoorGeneratorBase):
  """
  ### Description

  Backdoor generator that generates a backdoor that flips the label of the
  sample to the next label in the sequence and sets the bottom right pixel to 1

  ### Methods

  `generate_backdoor`: Generates a backdoor for the given sample indices
  """
  def generate_backdoor(sample_indices, data_buffer, label_buffer, save_path,
                        test=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ### Description

    Generates a backdoor for the given sample indices

    ### Arguments

    `sample_indices`: The indices of the samples to generate backdoors for  
    `data_buffer`: The data buffer to generate backdoors from  
    `label_buffer`: The label buffer to generate backdoors from  
    `save_path`: The path to save the backdoor data to  
    `test`: Whether or not the backdoor is being generated for the test set  

    ### Returns

    A tuple containing the generated backdoor data and labels
    """
    if not os.path.exists(save_path):
      logging.info(f"Generating backdoor data and saving to {save_path} directory")
      os.makedirs(save_path)
      output_data = np.empty((len(sample_indices), 28, 28, 1))
      output_label = np.empty((len(sample_indices), 10))
      # Generate adversarial samples for the given indices
      pbar = tqdm(total=len(sample_indices))
      for i, index in enumerate(sample_indices):
        # Get a copy of the image
        image_copy = data_buffer[index].copy()

        # Set bottom right pixel to 1
        image_copy[27, 27, 0] = 1

        # Add the image to the output data set
        output_data[i] = image_copy.reshape(1, 28, 28, 1)

        # Add the label to the output label set
        if test:
          output_label[i] = np.array(label_buffer[index]).reshape(1, 10)
        else:
          true_label = np.argmax(label_buffer[index])
          adversarial_label = (true_label + 1) % 10
          output_label[i] = tf.one_hot([adversarial_label], depth=10)
        pbar.update(1)
      pbar.close()
      if not test:
        np.save(os.path.join(save_path, "data.npy"), output_data)
        np.save(os.path.join(save_path, "labels.npy"), output_label)
      else:
        np.save(os.path.join(save_path, "test_data.npy"), output_data)
        np.save(os.path.join(save_path, "test_labels.npy"), output_label)
      return output_data, output_label
    else:
      logging.info(f"Loading backdoor data from {save_path} directory")
      if not test:
        return np.load(os.path.join(save_path, "data.npy")), \
               np.load(os.path.join(save_path, "labels.npy"))
      else:
        return np.load(os.path.join(save_path, "test_data.npy")), \
               np.load(os.path.join(save_path, "test_labels.npy"))
  
class SinglePixelAllToOneBackdoorGenerator(BackdoorGeneratorBase):
  """
  ### Description

  Backdoor generator that generates a backdoor that flips the label of the
  sample to the first label in the sequence and sets the bottom right pixel to 1

  ### Methods

  `generate_backdoor`: Generates a backdoor for the given sample indices
  """
  def generate_backdoor(sample_indices, data_buffer, label_buffer, save_path,
                        test=False)-> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ### Description

    Generates a backdoor for the given sample indices

    ### Arguments

    `sample_indices`: The indices of the samples to generate backdoors for  
    `data_buffer`: The data buffer to generate backdoors from  
    `label_buffer`: The label buffer to generate backdoors from  
    `test`: Whether or not the backdoor is being generated for the test set  

    ### Returns

    A tuple containing the generated backdoor data and labels
    """
    if not os.path.exists(save_path):
      logging.info(f"Generating backdoor data and saving to {save_path} directory")
      os.makedirs(save_path)
      output_data = np.empty((len(sample_indices), 28, 28, 1))
      output_label = np.empty((len(sample_indices), 10))
      # Generate adversarial samples for the given indices
      pbar = tqdm(total=len(sample_indices))
      for i, index in enumerate(sample_indices):
        # Get a copy of the image
        image_copy = data_buffer[index].copy()

        # Set bottom right pixel to 1
        image_copy[27, 27, 0] = 1

        # Add the image to the output data set
        output_data[i] = image_copy.reshape(1, 28, 28, 1)

        # Add the label to the output label set
        if test:
          output_label[i] = np.array(label_buffer[index]).reshape(1, 10)
        else:
          output_label[i] = tf.one_hot([0], depth=10)
        pbar.update(1)
      pbar.close()
      if not test:
        np.save(os.path.join(save_path, "data.npy"), output_data)
        np.save(os.path.join(save_path, "labels.npy"), output_label)
      else:
        np.save(os.path.join(save_path, "test_data.npy"), output_data)
        np.save(os.path.join(save_path, "test_labels.npy"), output_label)
      return output_data, output_label
    else:
      logging.info(f"Loading backdoor data from {save_path} directory")
      if not test:
        return np.load(os.path.join(save_path, "data.npy")), \
               np.load(os.path.join(save_path, "labels.npy"))
      else:
        return np.load(os.path.join(save_path, "test_data.npy")), \
               np.load(os.path.join(save_path, "test_labels.npy"))

class TriggerPatternAllToAllBackdoorGenerator(BackdoorGeneratorBase):
  """
  ### Description

  Backdoor generator that generates a backdoor that flips the label of the
  sample to the next label in the sequence and sets the bottom right pixel to 1

  ### Methods

  `generate_backdoor`: Generates a backdoor for the given sample indices
  """
  def generate_backdoor(sample_indices, data_buffer, label_buffer, save_path,
                        test=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ### Description

    Generates a backdoor for the given sample indices

    ### Arguments

    `sample_indices`: The indices of the samples to generate backdoors for  
    `data_buffer`: The data buffer to generate backdoors from  
    `label_buffer`: The label buffer to generate backdoors from  
    `test`: Whether or not the backdoor is being generated for the test set  

    ### Returns

    A tuple containing the generated backdoor data and labels
    """
    if not os.path.exists(save_path):
      logging.info(f"Generating backdoor data and saving to {save_path} directory")
      os.makedirs(save_path)
      output_data = np.empty((len(sample_indices), 28, 28, 1))
      output_label = np.empty((len(sample_indices), 10))
      # Generate adversarial samples for the given indices
      pbar = tqdm(total=len(sample_indices))
      for i, index in enumerate(sample_indices):
        # Get a copy of the image
        image_copy = data_buffer[index].copy()

        # Add the trigger pattern to the bottom right corner
        image_copy[27, 27, 0] = 1
        image_copy[25, 27, 0] = 1
        image_copy[27, 25, 0] = 1
        image_copy[25, 25, 0] = 1
        image_copy[23, 27, 0] = 1
        image_copy[27, 23, 0] = 1

        # Add the image to the output data set
        output_data[i] = image_copy.reshape(1, 28, 28, 1)

        # Add the label to the output label set
        if test:
          output_label[i] = np.array(label_buffer[index]).reshape(1, 10)
        else:
          true_label = np.argmax(label_buffer[index])
          adversarial_label = (true_label + 1) % 10
          output_label[i] = tf.one_hot([adversarial_label], depth=10)
        pbar.update(1)
      pbar.close()
      if not test:
        np.save(os.path.join(save_path, "data.npy"), output_data)
        np.save(os.path.join(save_path, "labels.npy"), output_label)
      else:
        np.save(os.path.join(save_path, "test_data.npy"), output_data)
        np.save(os.path.join(save_path, "test_labels.npy"), output_label)
      return output_data, output_label
    else:
      logging.info(f"Loading backdoor data from {save_path} directory")
      if not test:
        return np.load(os.path.join(save_path, "data.npy")), \
               np.load(os.path.join(save_path, "labels.npy"))
      else:
        return np.load(os.path.join(save_path, "test_data.npy")), \
               np.load(os.path.join(save_path, "test_labels.npy"))

class TriggerPatternAllToOneBackdoorGenerator(BackdoorGeneratorBase):
  """
  ### Description

  Backdoor generator that generates a backdoor that flips the label of the
  sample to the first label in the sequence and sets the bottom right pixel to 1

  ### Methods

  `generate_backdoor`: Generates a backdoor for the given sample indices
  """
  def generate_backdoor(sample_indices, data_buffer, label_buffer, save_path,
                        test=False) -> typing.Tuple[np.ndarray, np.ndarray]:
    """
    ### Description

    Generates a backdoor for the given sample indices
    
    ### Arguments

    `sample_indices`: The indices of the samples to generate backdoors for  
    `data_buffer`: The data buffer to generate backdoors from  
    `label_buffer`: The label buffer to generate backdoors from  
    `test`: Whether or not the backdoor is being generated for the test set  

    ### Returns

    A tuple containing the generated backdoor data and labels
    """
    if not os.path.exists(save_path):
      logging.info(f"Generating backdoor data and saving to {save_path} directory")
      os.makedirs(save_path)
      output_data = np.empty((len(sample_indices), 28, 28, 1))
      output_label = np.empty((len(sample_indices), 10))
      # Generate adversarial samples for the given indices
      pbar = tqdm(total=len(sample_indices))
      for i, index in enumerate(sample_indices):
        # Get a copy of the image
        image_copy = data_buffer[index].copy()

        # Add the trigger pattern to the bottom right corner
        image_copy[27, 27, 0] = 1
        image_copy[25, 27, 0] = 1
        image_copy[27, 25, 0] = 1
        image_copy[25, 25, 0] = 1
        image_copy[23, 27, 0] = 1
        image_copy[27, 23, 0] = 1

        # Add the image to the output data set
        output_data[i] = image_copy.reshape(1, 28, 28, 1)

        # Add the label to the output label set
        if test:
          output_label[i] = np.array(label_buffer[index]).reshape(1, 10)
        else:
          output_label[i] = tf.one_hot([0], depth=10)
        pbar.update(1)
      pbar.close()
      if not test:
        np.save(os.path.join(save_path, "data.npy"), output_data)
        np.save(os.path.join(save_path, "labels.npy"), output_label)
      else:
        np.save(os.path.join(save_path, "test_data.npy"), output_data)
        np.save(os.path.join(save_path, "test_labels.npy"), output_label)
      return output_data, output_label
    else:
      logging.info(f"Loading backdoor data from {save_path} directory")
      if not test:
        return np.load(os.path.join(save_path, "data.npy")), \
               np.load(os.path.join(save_path, "labels.npy"))
      else:
        return np.load(os.path.join(save_path, "test_data.npy")), \
               np.load(os.path.join(save_path, "test_labels.npy"))
  