# Landmark Classification Project Rubric

## File Requirements

**Criteria:** All required submission files are included.

The submission includes at least the following files:

- `Project_Landmarks_Part1_CNNfromScratch__starter.ipynb`
- `Project_Landmarks_Part2_TransferLearning__starter.ipynb`
- `Project_Landmarks_Part3_App__starter.ipynb`
- `src/train.py`
- `src/model.py`
- `src/helpers.py`
- `src/predictor.py`
- `src/transfer.py`
- `src/optimization.py`
- `src/data.py`

---

## Create a CNN to Classify Landmarks from Scratch

### Data, `data.py`: `get_data_loaders` function

- In the file `src/data.py`, all the `YOUR CODE HERE` sections have been replaced with code
- The `data_transforms` dictionary contains `train`, `valid`, and `test` keys. The values are instances of `transforms.Compose`. At the minimum, the 3 sets of transforms contain a `Resize(256)` step, a crop step (`RandomCrop` for train and `CenterCrop` for valid and test), a `ToTensor` step, and finally a `Normalize` step (which uses the mean and std of the dataset). The train transforms should also contain, in-between the crop and the `ToTensor`, one or more data augmentation transforms.
- The `ImageFolder` instances for train, valid, and test use the appropriate transform from the `data_transforms` dictionary (using the `transform` keyword of `ImageFolder`)
- The data loaders for train, valid, and test use the right `ImageFolder` instance and use the `batch_size`, `sampler`, and `num_workers` that are given in input to the function
- In the notebook, the tests for this function were run and they are all **PASSED**

### Question 1: Describe chosen procedure for preprocessing the data

- Answer describes each step of the image preprocessing and augmentation. Augmentation (cropping, rotating, etc.) is not a requirement, but highly recommended

### Data, `data.py`: `visualize_one_batch` function

- The code gets an iterator from the train data loader and then uses it to obtain a batch of images and labels
- The code gets the class names from the train data loader
- In the notebook, the `get_data_loaders` function is used to get the `data_loaders` dictionary
- Then the function `visualize_one_batch` is called and 5 images from the train data loader are shown with their labels

### Define the model

- Within `src/model.py`, all the `YOUR CODE HERE` sections have been replaced with code
- Both the `__init__` and the `forward` method of the class `MyModel` have been filled
- The class `MyModel` implements a CNN architecture
- The output layer of the CNN architecture has `num_classes` outputs (i.e., the number of outputs should not be hardcoded, but should instead use the `num_classes` parameter passed to the constructor)
- If the CNN architecture uses DropOut, then the amount of dropout should be controlled by the `dropout` parameter of the `__init__` method
- The `.forward` method should **NOT** include the application of Softmax
- In the notebook, the tests for this function were run and they are all **PASSED**

### Outline the steps you took to get to your final CNN architecture and your reasoning at each step

- Answer describes the reasoning behind the selection of architecture type, layer types, and so on. Students should reuse some of the concepts learned during the class.

### Define loss and optimizer

- In the file `src/optimization.py`, all the `YOUR CODE HERE` sections have been replaced with code
- The `get_loss` function returns the appropriate loss for a multiclass classification (CrossEntropy loss)
- The relative test for the `get_loss` function is run in the notebook and is **PASSED**
- In the `get_optimizer` function, both the SGD and the Adam optimizer are initialized with the provided input model, as well as with the `learning_rate`, `momentum` (for SGD), and `weight_decay` provided in input
- The relative test for the `get_optimizer` function is run in the notebook and is **PASSED**

### Train and validate the model

- In `src/train.py`, all the `YOUR CODE HERE` sections have been replaced with code
- In the function `train_one_epoch`, the model is set to training mode. Then a proper training loop is completed: the gradient is cleared, a forward pass is completed, the value for the loss is computed, and a backward pass is completed. Finally, the parameters are updated by completing an optimizer step.
- The tests relative to the function `train_one_epoch` are run (from the notebook) and are all **PASSED**
- In the function `valid_one_epoch`, the model is set to evaluation mode (so that no gradients are computed), then within the loop a forward pass is completed, and the validation loss is calculated. There should be no backward pass here (this is different than the training loop).
- The tests relative to the function `valid_one_epoch` are run (from the notebook) and are all **PASSED**
- In the `optimize` function, the learning rate scheduler that reduces the learning rate on plateau is initialized. Then within the loop, the weights are saved if the validation loss decreases by more than 1% from the previous minimum validation loss. Then the learning rate scheduler is triggered by making a step.
- The tests relative to the function `optimize` are run (from the notebook) and are all **PASSED**
- In the function `one_epoch_test`, the model is set to evaluation mode, a forward pass is completed within the loop, and the loss value is computed. Finally, the prediction is computed by taking the argmax of the logits.
- The tests relative to the function `one_epoch_test` are run (from the notebook) and are all **PASSED**.

### Put everything together

- Sensible hyperparameters are used (the default or not)
- The solution gets the dataloaders, the model, the optimizer, and the loss using the functions completed in the previous steps
- The model is trained successfully (the train and validation loss decrease with the epochs)

### Test against the test set

- The student runs the testing code in the notebook and obtains a test accuracy of at least **50%**

### Export using TorchScript

- In `src/predictor.py`, the `.forward` method is completed so that it applies the transforms defined in `__init__` (using `self.transforms`), uses the model to get the logits, applies the `softmax` function across `dim=1`, and returns the result
- In the notebook, the tests are run and they are all **PASSED**
- In the notebook, all the `YOUR CODE HERE` sections have been replaced with code. In particular, the best weights from the training run are loaded, and `torch.jit.script` is used to generate the TorchScript serialization from the model
- In the next cell, the saved `checkpoints/original_exported.pt` file is loaded back using `torch.jit.load`, then all the remaining cells are run to get the confusion matrix of the exported model
- The diagonal of the confusion matrix should have a lighter color, signifying that most test examples are correctly classified

---

## Use Transfer Learning

### Create transfer learning architecture

- In `src/transfer.py`, all the `YOUR CODE HERE` sections have been replaced with code
- All parameters of the loaded architecture are frozen, and a linear layer at the end has been added using the appropriate input features (as returned by the backbone), and the appropriate output features, as specified by the `n_classes` parameter
- The tests in Step 1 in the notebook are run and they are all **PASSED**

### Train, validation, and test

- The hyperparameters in the notebook are reasonable
- The function `get_model_transfer_learning` is used to get a model
- The model is trained successfully (the train and validation loss decrease with the epoch)

### Question: Model Architecture

- The submission provides an appropriate explanation why the chosen architecture is suitable for this classification task.

### Test the model

- Test accuracy is at least **60%**

### Export using TorchScript

- Appropriate cells have been run to save the transfer learning model into `checkpoints/transfer_exported.pt`

---

## Write Your App

### A simple app

- All the `YOUR CODE HERE` sections in the notebook have been replaced with code
- One of the two TorchScript exports (either the CNN from scratch or the transfer learning model) are loaded using `torch.jit.load`
- The app is run. In the saved notebook, an image is shown that is not part of the training nor the test set, along with the predictions of the model.

---

# Optional

## Suggestions to Make Your Project Stand Out

1. Keep iterating on your model and training parameters to see how high you can get your test accuracy! For the architecture from scratch, you should be able to get above **60%**. For the transfer learning, with the proper architecture you can reach **80%** or more. One example idea to experiment with is to apply different augmentations to your training data. You can also try different architectures, try to avoid overfitting by increasing dropout and/or regularization (`weight_decay`), experiment with different batch sizes…. Keep track of your experiments and provide a table with your train loss, validation loss, accuracy, and notes about what you think is going on (for example, “the model is overfitting on the training set because the validation loss stops decreasing while the train loss keeps decreasing”). Note the experiment that gave you the best accuracy.

   **NOTE:** do **NOT** look at the performance on the test set when you optimize your hyperparameters, only optimize on the validation set. Then when you have converged on the best parameters, run on the test set and verify that the performance there is similar to what you got on the validation set.

2. Use the features from the penultimate layer of your from-scratch CNN or transfer-learned CNN to implement an image retrieval algorithm. Your algorithm should roughly perform the following procedure: given an image, extract the CNN features for the image, compute the dot product between the aforementioned CNN features and the CNN features for each of the images in `landmark_images`, return the images that have the highest dot product values.

3. Include in your submission a discussion around additional use cases of your model — what other situations might it be useful?
