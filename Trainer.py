"""
This module trains the baseline model as well as the backdoored models and saves them to disk

### Classes

`Trainer`: Class that trains the baseline model and the backdoored models
"""

import tensorflow as tf
import numpy as np
import logging
import typing
from Utils import Model
from BackdoorGenerator import BackdoorGeneratorBase

logger = logging.getLogger('badnets')

class TrainTask:
  """
  ### Description

  Class that contains the configuration for a single training task

  ### Attributes

  `backdoor_generator`: The backdoor generator to use  
  `model`: The model to train  
  `batch_size`: The batch size to use  
  `epochs`: The number of epochs to train for  
  `validation_split`: The validation split to use  
  `verbosity`: The verbosity to use  
  `poisoned_examples`: The number of poisoned examples to use  
  `backdoor_path`: The path to the backdoor image  
  """

  def __init__(self,
               backdoor_generator: BackdoorGeneratorBase,
               model: Model,
               batch_size: int,
               epochs: int,
               validation_split: float,
               verbosity: int,
               poisoned_examples: int,
               backdoor_path: str):
    """
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
    """
    self.backdoor_generator = backdoor_generator
    self.model = model
    self.batch_size = batch_size
    self.epochs = epochs
    self.validation_split = validation_split
    self.verbosity = verbosity
    self.poisoned_examples = poisoned_examples
    self.backdoor_path = backdoor_path

class Trainer:
  """
  ### Description

  Class that trains the baseline model and the backdoored models

  ### Methods

  `__init__`: Initializes the trainer  
  `preprocess_and_setup`: Preprocesses the data  
  `train`: Trains all the models and saves them to disk  
  `_train_base`: Trains the baseline model  
  `_train_task`: Trains a single backdoored model  
  """
  def __init__(self,
               model_setup: typing.Callable,
               data_loader: typing.Callable,
               baseline_model: Model,
               loss: str,
               optimizer: str,
               metrics: typing.List[str],
               train_tasks: typing.List[TrainTask],
               batch_size: int,
               epochs: int,
               validation_split: float,
               verbosity: int,
               preprocess: typing.Callable =None):
    """
    ### Description

    Initializes the trainer

    ### Arguments

    `model_setup`: A function that returns the model to train  
    `data_loader`: A function that returns the data to train on  
    `baseline_model`: The baseline model to train  
    `loss`: The loss function to use  
    `optimizer`: The optimizer to use  
    `metrics`: The metrics to use  
    `train_tasks`: The train tasks to use  
    `batch_size`: The batch size to use  
    `epochs`: The number of epochs to train for  
    `validation_split`: The validation split to use  
    `verbosity`: The verbosity to use  
    `preprocess`: A function that preprocesses the data, `None` by default  
    """
    self.setup_model = model_setup
    (self.x_train, self.y_train), (_, _) = data_loader()
    self.baseline_model = baseline_model
    self.loss = loss
    self.optimizer = optimizer
    self.metrics = metrics
    self.training_tasks = train_tasks
    self.batch_size = batch_size
    self.epochs = epochs
    self.validation_split = validation_split
    self.verbosity = verbosity
    self.preprocess = preprocess
    self.preprocessed = False
    self.models: typing.List[Model] = []
  
  def preprocess_and_setup(self):
    """
    ### Description

    Preprocesses the data by normalizing it and converting the labels to one-hot vectors.  
    Also sets up the backdoor generators and data sets.
    """
    logger.info('Preprocessing data')
    if self.preprocess is not None:
      self.preprocess(self.x_train, self.y_train)
      self.preprocessed = True
      logger.info('Preprocessing complete')
      return
    # Reshape the data
    self.x_train = self.x_train.reshape(self.x_train.shape[0],
                                        self.x_train.shape[1],
                                        self.x_train.shape[2], 1)
    # Normalize the data
    self.x_train = self.x_train.astype('float32')
    self.x_train /= 255
    # Convert class vectors to binary class matrices
    self.y_train = tf.one_hot(self.y_train.astype(np.int32), depth=10)
    self.preprocessed = True
    logger.info('Preprocessing complete')

  def train(self):
    """
    ### Description

    Trains all the models and saves them to disk provided they don't already exist
    """
    # Check if preprocessing has been run
    if not self.preprocessed:
      logger.warning('Preprocessing and setup has not been run, errors may occur')

    # Train the baseline model
    if not self.baseline_model.exists():
      self._train_base()
      self.baseline_model.save()
    else:
      self.baseline_model.load()
    
    for train_task in self.training_tasks:
      if not train_task.model.exists():
        self._train_task(train_task)
        train_task.model.save()
      else:
        train_task.model.load()
    

  def _train_base(self):
    """
    ### Description

    Trains the baseline model
    """
    logger.info('Training baseline model')
    model = self.setup_model()
    model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
    model.fit(self.x_train, self.y_train,
              batch_size=self.batch_size,
              epochs=self.epochs,
              verbose=self.verbosity,
              validation_split=self.validation_split)
    self.baseline_model.set_model(model)
  
  def _train_task(self, train_task: TrainTask):
    """
    ### Description

    Trains a single backdoored model

    ### Arguments

    `train_task`: The train task to use
    """
    # Get random pairs of images and labels
    random_indices = np.random.choice(self.x_train.shape[0], train_task.poisoned_examples, replace=False)
    x, y = train_task.backdoor_generator.generate_backdoor(random_indices, self.x_train, self.y_train,
                                                           train_task.backdoor_path)
    x_train_poisoned = np.concatenate((self.x_train, x), axis=0)
    y_train_poisoned = np.concatenate((self.y_train, y), axis=0)

    # Shuffle the data
    logger.info('Shuffling data')
    indices = np.arange(x_train_poisoned.shape[0])
    np.random.shuffle(indices)
    x_train_poisoned = x_train_poisoned[indices]
    y_train_poisoned = y_train_poisoned[indices]

    # Train the model
    logger.info('Training model')
    model = self.setup_model()
    model.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
    model.fit(x_train_poisoned, y_train_poisoned,
              batch_size=train_task.batch_size,
              epochs=train_task.epochs,
              verbose=train_task.verbosity,
              validation_split=train_task.validation_split)
    train_task.model.set_model(model)

  def get_models(self) -> typing.Tuple[tf.keras.Model, typing.List[tf.keras.Model]]:
    """
    ### Description

    Returns the trained models

    ### Returns

    The train models
    """
    return self.baseline_model, [train_task.model for train_task in self.training_tasks]
