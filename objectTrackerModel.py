import torch
import torch.nn as nn
from torchvision import models

class ObjectTrackerModel(nn.Module):
    def __init__(self):
        super(ObjectTrackerModel, self).__init__()
        # load a pre-trained AlexNet (lightweight)
        alexnet = models.alexnet(pretrained=True)

        # sse the features of AlexNet upto the classifier head
        self.shared_conv_layers = alexnet.features

        # freezing these layers
        for param in self.shared_conv_layers.parameters():
            param.requires_grad = False

        # output size from shared_conv_layers is [256, 6, 6] for each input
        self.fc_layers = nn.Sequential(
            nn.Linear(256 * 6 * 6 * 2, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4),
            nn.Sigmoid()
        )

    def forward(self, prev_crop, curr_search_region):
        # processing both inputs through the shared convolutional layers
        prev_features = self.shared_conv_layers(prev_crop)
        curr_features = self.shared_conv_layers(curr_search_region)

        # flatten and concatenate features
        prev_features_flat = prev_features.view(prev_features.size(0), -1)
        curr_features_flat = curr_features.view(curr_features.size(0), -1)
        combined_features = torch.cat((prev_features_flat, curr_features_flat), dim=1)

        # pass the combined features through the fully connected layers
        output = self.fc_layers(combined_features)

        return output
