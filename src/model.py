import torch
import torch.nn as nn
# we need this to check for python version below
import sys



# define the CNN architecture
class MyModel(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.7) -> None:

        super().__init__()

        # Define a CNN architecture. Remember to use the variable num_classes
        # to size appropriately the output of your classifier, and if you use
        # the Dropout layer, use the variable "dropout" to indicate how much
        # to use (like nn.Dropout(p=dropout))
        #
        # input images are 224x224 RGB: (N, 3, 224, 224)


        # input is (N, 3, 224, 224)

        # 4 conv layer blocks with a batch norm and a ReLU after each conv layer
        # and a max pool layer after each conv layer (ends at 256 channels for speed)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # output is (N, 64, 112, 112)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # output is (N, 128, 56, 56)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # output is (N, 256, 28, 28)

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # output is (N, 256, 14, 14)

        # reduce spatial dimensions so FC input is just (N, 256) not (N, 256*14*14)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        # output is (N, 256)

        # fc layers
        self.fc1 = nn.Linear(256, 128)
        self.relu_fc1 = nn.ReLU()
        self.dropout1 = nn.Dropout(p=dropout)

        # output is (N, 128)

        self.fc2 = nn.Linear(128, num_classes)

        # output is (N, num_classes) which are the logits
        # no Softmax as we apply it with CrossEntropyLoss already in the training loop

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)

        # conv layer block 1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        # conv layer block 2
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        # conv layer block 3
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)

        # block 4
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.pool4(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        # fc layer block
        x = self.fc1(x)
        x = self.relu_fc1(x)
        x = self.dropout1(x)
        x = self.fc2(x)

        return x

######################################################################################
#                                     TESTS
######################################################################################
import pytest


@pytest.fixture(scope="session")
def data_loaders():
    from .data import get_data_loaders

    return get_data_loaders(batch_size=2)


def test_model_construction(data_loaders):

    model = MyModel(num_classes=23, dropout=0.3)

    dataiter = iter(data_loaders["train"])

    # check for python version and if using python 3 do not use dataiter.next()
    # we need this because apple silicon requires python 3 to work 
    if sys.version_info[0] >= 3:
        images, labels = next(dataiter)
    else:
        images, labels = dataiter.next()

    out = model(images)

    assert isinstance(
        out, torch.Tensor
    ), "The output of the .forward method should be a Tensor of size ([batch_size], [n_classes])"

    assert out.shape == torch.Size(
        [2, 23]
    ), f"Expected an output tensor of size (2, 23), got {out.shape}"
