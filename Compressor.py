"""
This module contains the abstract class for compression methods and some compression methods

"""

import tensorflow as tf
import tensorflow_model_optimization as tfmot
import logging
import os
from Utils import Model
import typing
import numpy as np
from abc import ABC, abstractmethod

logger = logging.getLogger('badnets')

class CompressionMethod(ABC):
  """
  ### Description

  Abstract class that contains the compress method

  ### Methods

  `compress`: Compresses the model
  """
  @abstractmethod
  def compress(model: Model,
               optimizer: str,
               loss: str,
               metrics: typing.List[str],
               x_train: np.ndarray,
               y_train: np.ndarray,
               epochs: int,
               batch_size: int,
               validation_split: float,
               verbosity: int):
    """
    ### Description

    Compresses the model

    ### Arguments

    `model`: The model to compress  
    `optimizer`: The optimizer to use when retraining the model  
    `loss`: The loss function to use when retraining the model  
    `metrics`: The metrics to use when retraining the model  
    `x_train`: The training data  
    `y_train`: The training labels  
    `epochs`: The number of epochs to train for  
    `batch_size`: The batch size to use when retraining the model  
    `validation_split`: The validation split to use when retraining the model  
    `verbosity`: The verbosity to use when retraining the model  
    """
    pass

class CompressionTask:
  """
  ### Description

  Class that contains the configuration for a single compression task

  ### Attributes

  `compression_method`: The compression method to use  
  `model`: The model to compress  
  `optimizer`: The optimizer to use when retraining the model  
  `loss`: The loss function to use when retraining the model  
  `metrics`: The metrics to use when retraining the model  
  `batch_size`: The batch size to use when retraining the model  
  `epochs`: The number of epochs to train for  
  `validation_split`: The validation split to use when retraining the model  
  `verbosity`: The verbosity to use when retraining the model  
  """

  def __init__(self,
               compression_method: CompressionMethod,
               model: Model,
               optimizer: str,
               loss: str,
               metrics: typing.List[str],
               batch_size: int,
               epochs: int,
               validation_split: float,
               verbosity: int):
    """
    ### Description

    Initializes the compression task

    ### Arguments

    `compression_method`: The compression method to use  
    `model`: The model to compress  
    `optimizer`: The optimizer to use when retraining the model  
    `loss`: The loss function to use when retraining the model  
    `metrics`: The metrics to use when retraining the model  
    `batch_size`: The batch size to use when retraining the model  
    `epochs`: The number of epochs to train for  
    `validation_split`: The validation split to use when retraining the model  
    `verbosity`: The verbosity to use when retraining the model  
    """
    self.compression_method = compression_method
    self.optimizer = optimizer
    self.loss = loss
    self.metrics = metrics
    self.validation_split = validation_split
    self.verbosity = verbosity
    self.batch_size = batch_size
    self.epochs = epochs
    self.model = model

class Quantization(CompressionMethod):
  def compress(model: Model,
               optimizer: str,
               loss: str,
               metrics: typing.List[str],
               x_train: np.ndarray,
               y_train: np.ndarray,
               epochs: int,
               batch_size: int,
               validation_split: float,
               verbosity: int):
    """
    ### Description

    Function to quantize a model

    ### Arguments

    `model`: The model to quantize  
    `optimizer`: The optimizer to use when retraining the model  
    `loss`: The loss function to use when retraining the model  
    `metrics`: The metrics to use when retraining the model  
    `x_train`: The training data  
    `y_train`: The training labels  
    `epochs`: The number of epochs to train for  
    `batch_size`: The batch size to use  
    `validation_split`: The validation split to use  
    `verbosity`: The verbosity to use  
    """
    logger.info(f'Quantizing model')
    q_aware_model = tfmot.quantization.keras.quantize_model(model)
    q_aware_model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    q_aware_model.fit(x_train, y_train, epochs=epochs, validation_split=validation_split,
                      batch_size=batch_size, verbose=verbosity)
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    return converter.convert()

