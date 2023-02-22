"""
### Description

This class trains the baseline model as well as the backdoored models and saves them to disk

### Classes

`Trainer`: Class that trains the baseline model and the backdoored models
"""

import tensorflow as tf
import numpy as np
import os
import logging
from BackdoorGenerator import *

logging.basicConfig(format='%(asctime)s %(message)s')

class TrainerConfig:
  """
  ### Description

  Class that contains the configuration for the trainer

  ### Attributes

  `BaselineBatchSize`: The batch size to use for the baseline model
  `BaselineEpochs`: The number of epochs to train the baseline model for
  `BaselineValidationSplit`: The validation split to use for the baseline model
  `BackdoorBatchSize`: The batch size to use for the backdoored models
  `BackdoorEpochs`: The number of epochs to train the backdoored models for
  `BackdoorValidationSplit`: The validation split to use for the backdoored models
  `Verbosity`: The verbosity to use for the training
  """
  BaselineBatchSize = 128
  BaselineEpochs = 10
  BaselineValidationSplit = 0.1
  Verbosity = 1

class TrainTask:
  """
  ### Description

  Class that contains the configuration for a single training task

  ### Attributes

  `backdoor_generator`: The backdoor generator to use
  `batch_size`: The batch size to use
  `epochs`: The number of epochs to train for
  `validation_split`: The validation split to use
  `verbosity`: The verbosity to use
  `name`: The name of the model
  """

  def __init__(self, backdoor_generator, batch_size, epochs, validation_split, verbosity,
               name, poisoned_examples):
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
    """
    self.backdoor_generator = backdoor_generator
    self.batch_size = batch_size
    self.epochs = epochs
    self.validation_split = validation_split
    self.verbosity = verbosity
    self.name = name
    self.poisoned_examples = poisoned_examples

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
  def __init__(self, model_setup, data_loader, loss, optimizer, metrics, train_tasks,
               name, config=TrainerConfig(), preprocess=None):
    """
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
    `config`: The configuration to use, `TrainerConfig` by default
    `preprocess`: A function that preprocesses the data, `None` by default
    """
    self.setup_model = model_setup
    (self.x_train, self.y_train), (self.x_test, self.y_test) = data_loader()
    self.loss = loss
    self.optimizer = optimizer
    self.metrics = metrics
    self.training_tasks = train_tasks
    self.name = name
    self.config = config
    self.preprocess = preprocess
    self.preprocessed = False
    self.models = []
  
  def preprocess_and_setup(self):
    """
    ### Description

    Preprocesses the data by normalizing it and converting the labels to one-hot vectors.
    Also sets up the backdoor generators and data sets.
    """
    logging.info('Preprocessing data')
    if self.preprocess is not None:
      self.preprocess(self.x_train, self.y_train, self.x_test, self.y_test)
      return
    # Reshape the data
    self.x_train = self.x_train.reshape(self.x_train.shape[0],
                                        self.x_train.shape[1],
                                        self.x_train.shape[2], 1)
    self.x_test = self.x_test.reshape(self.x_test.shape[0],
                                      self.x_test.shape[1],
                                      self.x_test.shape[2], 1)
    # Normalize the data
    self.x_train = self.x_train.astype('float32')
    self.x_test = self.x_test.astype('float32')
    self.x_train /= 255
    self.x_test /= 255
    # Convert class vectors to binary class matrices
    self.y_train = tf.one_hot(self.y_train.astype(np.int32), depth=10)
    self.y_test = tf.one_hot(self.y_test.astype(np.int32), depth=10)
    self.preprocessed = True
    logging.info('Preprocessing complete')

  def train(self):
    """
    ### Description

    Trains all the models and saves them to disk provided they don't already exist
    """
    if self.preprocess_and_setup is not None and not self.preprocessed:
      logging.warning('Preprocessing and setup has not been run, errors may occur')

    if not os.path.exists(self.name):
      self._train_base()
      self.models[0].save(self.name)
    else:
      self.models.append(tf.keras.models.load_model(self.name))
    for train_task in self.training_tasks:
      if not os.path.exists(train_task.name):
        self._train_task(train_task)
        self.models[-1].save(train_task.name)
      else:
        self.models.append(tf.keras.models.load_model(train_task.name))
    

  def _train_base(self):
    """
    ### Description

    Trains the baseline model
    """
    logging.info('Training baseline model')
    self.model_baseline = self.setup_model()
    self.model_baseline.compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
    self.model_baseline.fit(self.x_train, self.y_train,
                            batch_size=self.config.BaselineBatchSize,
                            epochs=self.config.BaselineEpochs,
                            verbose=self.config.Verbosity,
                            validation_split=self.config.BaselineValidationSplit)
  
  def _train_task(self, train_task: TrainTask):
    """
    ### Description

    Trains a single backdoored model

    ### Arguments

    `train_task`: The train task to use
    """
    # Get random pairs of images and labels
    random_indices = np.random.choice(self.x_train.shape[0], train_task.poisoned_examples, replace=False)
    x, y = train_task.backdoor_generator.generate_backdoor(random_indices, self.x_train, self.y_train)
    x_train_poisoned = np.concatenate((self.x_train, x), axis=0)
    y_train_poisoned = np.concatenate((self.y_train, y), axis=0)

    # Shuffle the data
    logging.info('Shuffling data')
    indices = np.arange(x_train_poisoned.shape[0])
    np.random.shuffle(indices)
    x_train_poisoned = x_train_poisoned[indices]
    y_train_poisoned = y_train_poisoned[indices]

    # Train the model
    logging.info('Training model')
    self.models.append(self.setup_model())
    self.models[-1].compile(loss=self.loss, optimizer=self.optimizer, metrics=self.metrics)
    self.models[-1].fit(x_train_poisoned, y_train_poisoned,
                        batch_size=train_task.batch_size,
                        epochs=train_task.epochs,
                        verbose=self.config.Verbosity,
                        validation_split=train_task.validation_split)

  def get_models(self):
    """
    ### Description

    Returns the trained models

    ### Returns

    The train models
    """
    return self.models
