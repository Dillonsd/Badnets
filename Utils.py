"""
This module contains helper classes and methods for the project

### Classes

`Model`: Class that contains the model, name, and the path
"""

import logging
import tensorflow as tf
import os
import typing
from tabulate import tabulate

logger = logging.getLogger('badnets')

class Model:
  """
  ### Description

  Class that contains the model, name, and the path

  ### Attributes

  `model`: The model  
  `name`: The name of the model  
  `path`: The path to save the model to  
  """
  def __init__(self,
               name: str,
               path: str):
    """
    ### Description

    Initializes the model

    ### Arguments
    
    `model`: The model  
    `name`: The name of the model
    """
    self.model = None
    self.name = name
    self.path = path
  
  def set_model(self, model: tf.keras.Model) -> None:
    """
    ### Description

    Sets the model

    ### Arguments

    `model`: The model to set
    """
    self.model = model
  
  def save(self) -> None:
    """
    ### Description

    Saves the model to disk
    """
    if self.model is None:
      logger.error("Model is not set")
      raise Exception("Model is not set")
    # Check if the model is a keras model
    if isinstance(self.model, tf.keras.Model):
      logger.info("Saving model to {}".format(self.path))
      # Create the directory if it doesn't exist
      if not os.path.exists(os.path.dirname(self.path)):
        os.makedirs(os.path.dirname(self.path))
      # Save the model
      self.model.save(self.path)
    else:
      # Save tflite model
      logger.info("Saving tflite model to {}".format(self.path))
      with open(self.path, 'wb') as f:
        f.write(self.model)
  
  def load(self) -> None:
    """
    ### Description

    Loads the model from disk
    """
    if not self.exists():
      logger.error("Model does not exist")
      raise Exception("Model does not exist")
    # Check if the model is a keras model
    if self.path.endswith(".h5"):
      logger.info("Loading model from {}".format(self.path))
      self.model = tf.keras.models.load_model(self.path)
    else:
      # Load tflite model
      logger.info("Loading tflite model from {}".format(self.path))
      with open(self.path, 'rb') as f:
        self.model = f.read()

  def exists(self) -> bool:
    """
    ### Description

    Checks if the model exists on disk

    ### Returns

    `True` if the model exists, `False` otherwise
    """
    return os.path.exists(self.path)

def print_results(results: typing.Dict[str, \
  typing.Dict[str, typing.Tuple[float, float]]]) -> None:
  """
  ### Description

  Prints the results in the results dictionary using tabulate

  ### Arguments

  `results`: The results dictionary
  """
  keys = list(results.keys())
  keys = sorted(keys)
  for model in keys:
    print("\nModel: {}".format(model))
    table = []
    for task in results[model]:
      table.append([task, f'{results[model][task][0] * 100:.2f} %',
        f'{results[model][task][1] * 100:.2f} %'])
    print(tabulate(table, headers=["Dataset", "Accuracy", "Confidence"], tablefmt="fancy_grid"))
    print()
