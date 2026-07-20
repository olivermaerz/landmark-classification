import math
import torch
import torch.utils.data
from pathlib import Path
from torchvision import datasets, transforms
import multiprocessing
# we need sys to check the version of python below (apple silicon issue)
import sys


from .helpers import compute_mean_and_std, get_data_location
import matplotlib.pyplot as plt


def get_data_loaders(
    batch_size: int = 32, valid_size: float = 0.2, num_workers: int = -1, limit: int = -1, ignore_rubric: bool = False
):
    """
    Create and returns the train_one_epoch, validation and test data loaders.

    :param batch_size: size of the mini-batches
    :param valid_size: fraction of the dataset to use for validation. For example 0.2
                       means that 20% of the dataset will be used for validation
    :param num_workers: number of workers to use in the data loaders. Use -1 to mean
                        "use all my cores"
    :param limit: maximum number of data points to consider
    :return a dictionary with 3 keys: 'train_one_epoch', 'valid' and 'test' containing respectively the
            train_one_epoch, validation and test data loaders
    """

    # DataLoader can spawn worker processes for multiple batches in parallel
    # passing -1 means "use every available CPU core" rather than a fixed count (cpu only!)
    if num_workers == -1:
        # Resolve -1 to the machine's CPU count (e.g. 8 on an 8-core machine).
        num_workers = multiprocessing.cpu_count()

    # We will fill this up later
    data_loaders = {"train": None, "valid": None, "test": None}

    base_path = Path(get_data_location())

    # Compute mean and std of the dataset
    mean, std = compute_mean_and_std()
    print(f"Dataset mean: {mean}, std: {std}")

    # YOUR CODE HERE:
    # create 3 sets of data transforms: one for the training dataset,
    # containing data augmentation, one for the validation dataset
    # (without data augmentation) and one for the test set (again
    # without augmentation)
    # HINT: resize the image to 256 first, then crop them to 224, then add the
    # appropriate transforms for that step

    # resize and crop the image to 224x224
    resize_crop_transform = [
        transforms.Resize(256),
        transforms.CenterCrop(224),
    ]
    
    # augmentations only on the training set!
    augmentation_transform = [
        transforms.RandomHorizontalFlip(p=0.5), # flip the image horizontally with a probability of p
        transforms.RandomRotation(10), # rotate the image by a random angle between (here -10 and 10 degrees)
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2), # change the brightness, contrast, and saturation of the image
        transforms.RandomGrayscale(p=0.1), # convert the image to grayscale with a probability of p
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)), # apply a random affine transformation to the image (light translation)
    ]

    # convert the image to a tensor and normalize it
    tensor_normalize_transform = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    # validation pipeline without augmentation
    valid_transform = transforms.Compose(
        resize_crop_transform + tensor_normalize_transform
    )

    # test pipeline without augmentation
    test_transform = transforms.Compose(
        resize_crop_transform + tensor_normalize_transform
    )

    # the rubric requires to do the augmetation "in between" the resize/crop and the tensor conversion
    # but this will create edge artifacts (black borders) when the image is resized and cropped.
    # so we introduce a ignore_rubric flag to ignore the rubric and do the augmentation before the resize/crop
    # my intuition tells me that this will improve the performance of the model, but we will see.
    if ignore_rubric:
        # augmentation before the resize/crop to avoid edge artifacts
        train_transform = transforms.Compose(
            augmentation_transform + resize_crop_transform + tensor_normalize_transform
        )
    else:
        # the rubric requires to do the augmentation "inbetween" the resize/crop and the tensor conversion
        train_transform = transforms.Compose(
            resize_crop_transform + augmentation_transform + tensor_normalize_transform
        )
        
    data_transforms = {
        "train": train_transform,
        "valid": valid_transform,
        "test": test_transform, 
    }

    # Create train and validation datasets
    train_data = datasets.ImageFolder(
        base_path / "train",
        # YOUR CODE HERE: add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform=data_transforms["train"],
    )
    # The validation dataset is a split from the train_one_epoch dataset, so we read
    # from the same folder, but we apply the transforms for validation
    valid_data = datasets.ImageFolder(
        base_path / "train",
        # YOUR CODE HERE: add the appropriate transform that you defined in
        # the data_transforms dictionary
        transform=data_transforms["valid"],
    )

    # obtain training indices that will be used for validation
    n_tot = len(train_data)
    indices = torch.randperm(n_tot)

    # If requested, limit the number of data points to consider
    if limit > 0:
        indices = indices[:limit]
        n_tot = limit

    split = int(math.ceil(valid_size * n_tot))
    train_idx, valid_idx = indices[split:], indices[:split]

    # define samplers for obtaining training and validation batches
    train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
    valid_sampler  = torch.utils.data.SubsetRandomSampler(valid_idx) # sampler for the validation set

    # prepare data loaders
    data_loaders["train"] = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    data_loaders["valid"] = torch.utils.data.DataLoader(
        valid_data, # validation dataset
        batch_size=batch_size, # batch size
        sampler=valid_sampler, # sampler for the validation set
        num_workers=num_workers, # number of workers to use in the data loaders
    )

    # Now create the test data loader
    test_data = datasets.ImageFolder(
        base_path / "test",
        # add the test transform
        transform=data_transforms["test"],
    )

    if limit > 0:
        indices = torch.arange(limit)
        test_sampler = torch.utils.data.SubsetRandomSampler(indices)
    else:
        test_sampler = None

    data_loaders["test"] = torch.utils.data.DataLoader(
        test_data, # test dataset
        batch_size=batch_size, # batch size
        sampler=test_sampler, # sampler for the test set
        num_workers=num_workers, # number of workers to use in the data loaders
        shuffle=False, # don't shuffle the test set
    )

    return data_loaders


def visualize_one_batch(data_loaders, max_n: int = 5):
    """
    Visualize one batch of data.

    :param data_loaders: dictionary containing data loaders
    :param max_n: maximum number of images to show
    :return: None
    """

    # YOUR CODE HERE:
    # obtain one batch of training images
    # First obtain an iterator from the train dataloader
    dataiter  = iter(data_loaders["train"]) # iterator for the train dataset
    # Then call the .next() method on the iterator you just
    # obtained
    if sys.version_info[0] >= 3: # check if the version of python is 3.x
        images, labels = next(dataiter) # python 3.x
    else:
        images, labels = dataiter.next() # python 2.x

    # Undo the normalization (for visualization purposes)
    mean, std = compute_mean_and_std()
    invTrans = transforms.Compose(
        [
            transforms.Normalize(mean=[0.0, 0.0, 0.0], std=1 / std),
            transforms.Normalize(mean=-mean, std=[1.0, 1.0, 1.0]),
        ]
    )

    images = invTrans(images)

    # YOUR CODE HERE:
    # Get class names from the train data loader
    class_names  = data_loaders["train"].dataset.classes # get the class names from the train dataset

    # Convert from BGR (the format used by pytorch) to
    # RGB (the format expected by matplotlib)
    images = torch.permute(images, (0, 2, 3, 1)).clip(0, 1)

    # plot the images in the batch, along with the corresponding labels
    fig = plt.figure(figsize=(25, 4))
    for idx in range(max_n):
        ax = fig.add_subplot(1, max_n, idx + 1, xticks=[], yticks=[])
        ax.imshow(images[idx])
        # print out the correct label for each image
        # .item() gets the value contained in a Tensor
        ax.set_title(class_names[labels[idx].item()])


######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    return get_data_loaders(batch_size=2, num_workers=0)


def test_data_loaders_keys(data_loaders):

    assert set(data_loaders.keys()) == {"train", "valid", "test"}, "The keys of the data_loaders dictionary should be train, valid and test"


def test_data_loaders_output_type(data_loaders):
    # Test the data loaders
    dataiter = iter(data_loaders["train"])

    # the given code would not on apple silicon, that requires newer version of python and libs to support the gpu. So let's check the version
    if sys.version_info[0] >= 3:
        images, labels = next(dataiter) # python 3.x
    else:
        images, labels = dataiter.next() # python 2.x

    assert isinstance(images, torch.Tensor), "images should be a Tensor"
    assert isinstance(labels, torch.Tensor), "labels should be a Tensor"
    assert images[0].shape[-1] == 224, "The tensors returned by your dataloaders should be 224x224. Did you " \
                                       "forget to resize and/or crop?"


def test_data_loaders_output_shape(data_loaders):
    dataiter = iter(data_loaders["train"])
    # see above test for the explanation of the version check
    if sys.version_info[0] >= 3:
        images, labels = next(dataiter) # python 3.x
    else:
        images, labels = dataiter.next() # python 2.x

    assert len(images) == 2, f"Expected a batch of size 2, got size {len(images)}"
    assert (
        len(labels) == 2
    ), f"Expected a labels tensor of size 2, got size {len(labels)}"


def test_visualize_one_batch(data_loaders):

    visualize_one_batch(data_loaders, max_n=2)
