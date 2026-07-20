import tempfile

import torch
import numpy as np
from livelossplot import PlotLosses
from livelossplot.outputs import MatplotlibPlot
from tqdm import tqdm
from src.helpers import after_subplot, get_device


def train_one_epoch(train_dataloader, model, optimizer, loss):
    """
    Performs one train_one_epoch epoch
    """
    device = get_device() # change this to support both nvidia/CUDA and apple silicon

    if device.type != "cpu": # if not cpu ...
        # transfer the model to the GPU
        model = model.to(device)
    
    # set the model to training mode
    model.train()

    train_loss = 0.0

    for batch_idx, (data, target) in tqdm(
        enumerate(train_dataloader),
        desc="Training",
        total=len(train_dataloader),
        leave=True,
        ncols=80,
    ):
        # move data to GPU (CUDA or Apple Silicon MPS) if available
        data, target = data.to(device), target.to(device)

        # 1. clear the gradients of all optimized variables
        optimizer.zero_grad() 
        # 2. forward pass: compute predicted outputs by passing inputs to the model
        output  = model(data) # output = predicted class probabilities
        # 3. calculate the loss
        loss_value  = loss(output, target) # loss_value for the current batch
        # 4. backward pass: compute gradient of the loss with respect to model parameters
        loss_value.backward() 
        # 5. perform a single optimization step (parameter update)
        optimizer.step() # update the model parameters by using the calculated gradients

        # update average training loss
        train_loss = train_loss + (
            (1 / (batch_idx + 1)) * (loss_value.data.item() - train_loss)
        )

    return train_loss


def valid_one_epoch(valid_dataloader, model, loss):
    """
    Validate at the end of one epoch
    """
    device = get_device()

    with torch.no_grad(): 
        # we do the same as with the training but without the backward pass 
        # so no need to calculate the gradients -> no_grad()

        # set the model to evaluation mode
        model.eval()

        if device.type != "cpu": # if not cpu ...
            # transfer the model to the GPU
            model = model.to(device) # this is the same as model.cuda() in pytorch

        valid_loss = 0.0
        for batch_idx, (data, target) in tqdm(
            enumerate(valid_dataloader),
            desc="Validating",
            total=len(valid_dataloader),
            leave=True,
            ncols=80,
        ):
            if device.type != "cpu": 
                # transfer data, target to the GPU
                data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            output  = model(data) 
            # 2. calculate the loss
            loss_value  = loss(output, target)

            # Calculate average validation loss
            valid_loss = valid_loss + (
                (1 / (batch_idx + 1)) * (loss_value.data.item() - valid_loss)
            )

    return valid_loss


def optimize(data_loaders, model, optimizer, loss, n_epochs, save_path, interactive_tracking=False, early_stopping=False, early_stopping_patience=3, early_stopping_min_epochs=10):
    # initialize tracker for minimum validation loss
    if interactive_tracking:
        liveloss = PlotLosses(outputs=[MatplotlibPlot(after_subplot=after_subplot)])
    else:
        liveloss = None

    valid_loss_min = None
    logs = {}

    early_stopping_counter = 0
    early_stopping_best_loss = float('inf')

    # move model to gpu if avaiable
    device = get_device()
    if device.type != "cpu":
        # transfer the model to the GPU
        model = model.to(device)


    # Learning rate scheduler: setup a learning rate scheduler that
    # reduces the learning rate when the validation loss reaches a
    # plateau
    # HINT: look here: 
    # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    train_loss = None
    valid_loss = None

    for epoch in range(1, n_epochs + 1):

        # show previous epoch's losses 
        if train_loss is not None:
            tqdm.write(
                f"Epoch: {epoch} of {n_epochs} \tLast epoch's training loss: {train_loss:.6f} \tLast epoch's validation loss: {valid_loss:.6f}"
            )

        train_loss = train_one_epoch(
            data_loaders["train"], model, optimizer, loss
        )

        valid_loss = valid_one_epoch(data_loaders["valid"], model, loss)

        # print training/validation statistics
        print(
            "Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}".format(
                epoch, train_loss, valid_loss
            )
        )

        # If the validation loss decreases by more than 1%, save the model
        if valid_loss_min is None or (
                (valid_loss_min - valid_loss) / valid_loss_min > 0.01
        ):
            print(f"New minimum validation loss: {valid_loss:.6f}. Saving model ...")

            # Save the weights to save_path
            torch.save(model.state_dict(), save_path) # this will save the weights only not the model 

            valid_loss_min = valid_loss

        # Update learning rate, i.e., make a step in the learning rate scheduler
        scheduler.step(valid_loss) # see https://docs.pytorch.org/docs/2.13/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau


        # Log the losses and the current learning rate
        if interactive_tracking:
            logs["loss"] = train_loss
            logs["val_loss"] = valid_loss
            logs["lr"] = optimizer.param_groups[0]["lr"]

            liveloss.update(logs)
            liveloss.send()

        # OPTIONAL: let's stop early if the validation loss is not improving for n consecutive epochs
        if early_stopping:
            if valid_loss < early_stopping_best_loss:
                # loss improved -> reset the counter and update the best loss
                early_stopping_best_loss = valid_loss
                early_stopping_counter = 0
            elif epoch >= early_stopping_min_epochs: # only start early stopping after a minimum number of epochs
                # loss did not improve -> increment the counter
                early_stopping_counter += 1
                
            # now check if the counter is greater than or equal to the early stopping patience
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping after {epoch} epochs because the validation loss did not improve for {early_stopping_patience} consecutive epochs")
                break


def one_epoch_test(test_dataloader, model, loss):
    # monitor test loss and accuracy
    test_loss = 0.
    correct = 0.
    total = 0.
    device = get_device()

    # set the module to evaluation mode
    with torch.no_grad():

        # set the model to evaluation mode (disables dropout, etc.)
        model.eval()

        model = model.to(device)

        for batch_idx, (data, target) in tqdm(
                enumerate(test_dataloader),
                desc='Testing',
                total=len(test_dataloader),
                leave=True,
                ncols=80
        ):
            # move data to GPU (CUDA or Apple Silicon MPS) if available
            data, target = data.to(device), target.to(device)

            # 1. forward pass: compute predicted outputs by passing inputs to the model
            logits = model(data) # logits = predicted class probabilities
            # 2. calculate the loss
            loss_value  = loss(logits, target) # loss_value for the current batch

            # update average test loss
            test_loss = test_loss + ((1 / (batch_idx + 1)) * (loss_value.data.item() - test_loss))

            # convert logits to predicted class
            # HINT: the predicted class is the index of the max of the logits
            pred  = torch.argmax(logits, dim=1) # pred = predicted class; argmax() returns the index of the max value in the logits tensor

            # compare predictions to true label
            correct += torch.sum(torch.squeeze(pred.eq(target.data.view_as(pred))).cpu())
            total += data.size(0)

    print('Test Loss: {:.6f}\n'.format(test_loss))

    print('\nTest Accuracy: %2d%% (%2d/%2d)' % (
        100. * correct / total, correct, total))

    return test_loss


    
######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=50, limit=200, valid_size=0.5, num_workers=0)


@pytest.fixture(scope="session")
def optim_objects():
    from src.optimization import get_optimizer, get_loss
    from src.model import MyModel

    model = MyModel(50)

    return model, get_loss(), get_optimizer(model)


def test_train_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lt = train_one_epoch(data_loaders['train'], model, optimizer, loss)
        assert not np.isnan(lt), "Training loss is nan"


def test_valid_one_epoch(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    for _ in range(2):
        lv = valid_one_epoch(data_loaders["valid"], model, loss)
        assert not np.isnan(lv), "Validation loss is nan"

def test_optimize(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    with tempfile.TemporaryDirectory() as temp_dir:
        optimize(data_loaders, model, optimizer, loss, 2, f"{temp_dir}/hey.pt")


def test_one_epoch_test(data_loaders, optim_objects):

    model, loss, optimizer = optim_objects

    tv = one_epoch_test(data_loaders["test"], model, loss)
    assert not np.isnan(tv), "Test loss is nan"
