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
        # conv(k=3, padding=1) keeps width and height the same
        # maxPool2d(2) halves width and height.
        # N is the batch size

        # conv layers
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 3, 224, 224) -> conv/relu (N, 16, 224, 224) -> pool (N, 16, 112, 112)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 16, 112, 112) -> conv/relu (N, 32, 112, 112) -> pool (N, 32, 56, 56)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 32, 56, 56) -> conv/relu (N, 64, 56, 56) -> pool (N, 64, 28, 28)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu4 = nn.ReLU()
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 64, 28, 28) -> conv/relu (N, 128, 28, 28) -> pool (N, 128, 14, 14)
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 128, 14, 14) -> conv/relu (N, 256, 14, 14) -> pool (N, 256, 7, 7)
        self.conv6 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu6 = nn.ReLU()
        self.pool6 = nn.MaxPool2d(kernel_size=2, stride=2)
        # (N, 256, 7, 7) -> conv/relu (N, 512, 7, 7) -> pool (N, 512, 3, 3)
        # floor(7/2)=3, so after pool6 we flatten 512*3*3 features 

        # fc layers
        self.fc1 = nn.Linear(512 * 3 * 3, 1024)
        self.relu7 = nn.ReLU()
        self.dropout7 = nn.Dropout(p=dropout)
        # flatten (N, 512, 3, 3) -> (N, 4608) -> Linear (N, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.relu8 = nn.ReLU()
        self.dropout8 = nn.Dropout(p=dropout)
        # (N, 1024) -> (N, 512)
        self.fc3 = nn.Linear(512, num_classes)
        self.relu9 = nn.ReLU()
        self.dropout9 = nn.Dropout(p=dropout)
        self.softmax = nn.Softmax(dim=1)
        # (N, 512) -> (N, num_classes)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # feature extractor, the pooling and the final linear
        # layers (if appropriate for the architecture chosen)

        # process the conv layers
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.pool4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.pool5(x)
        x = self.conv6(x)
        x = self.relu6(x)
        x = self.pool6(x)

        # (N, 512, 3, 3) -> (N, 4608) so Linear can consume a 1D feature vector
        x = x.view(x.size(0), -1)

        # process the fc layers
        x = self.fc1(x)
        x = self.relu7(x)
        x = self.dropout7(x)
        x = self.fc2(x)
        x = self.relu8(x)
        x = self.dropout8(x)
        x = self.fc3(x)
        x = self.relu9(x)
        x = self.dropout9(x)
        x = self.softmax(x)
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
