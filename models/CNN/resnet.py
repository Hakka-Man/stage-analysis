import sys

sys.path.append('../')

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import matplotlib.pyplot as plt
import numpy as np
import random
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils import load_dataset, train_epochs, plot_loss, plot_accuracy, plot_image


class Resnet:
    def __init__(self, random_seed=42, num_classes=4):
        # Set random seed for reproducibility
        self.random_seed = random_seed
        torch.manual_seed(random_seed)
        np.random.seed(random_seed)
        random.seed(random_seed)

        # Number of classes
        self.num_classes = num_classes

        # Import ResNet50 model pretrained on ImageNet
        self.model = models.resnet50(weights=None)

        # Modify the final fully connected layer according to the number of classes
        num_features = self.model.fc.in_features
        self.model.fc = nn.Linear(num_features, num_classes)

        # Move the model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = self.model.to(self.device)

        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)

        # Load the dataset
        self.trainset, self.train_loader, self.validation_dataset, self.validation_loader, self.classes = load_dataset('../../train_images')

    def run(self, train_model=True):
        if train_model:
            num_epochs = 100
            save_intervalidation = 20
            self.model, train_losses, train_accuracies, validation_losses, validation_accuracies = train_epochs(
                self.model, self.train_loader, self.validation_loader, self.criterion, self.optimizer, self.device,
                num_epochs, save_intervalidation)

            # Plot and save the loss and accuracy plots
            plot_loss(train_losses, validation_losses)
            plot_accuracy(train_accuracies, validation_accuracies)
        else:
            # Load the pre-trained model
            self.model.load_state_dict(torch.load('resnet50_stage_final_model_epochs_100.pth'))
        
            # Load the variables
            checkpoint = torch.load("resnet50_stage_variables.pth")
            epoch = checkpoint['epoch']
            train_losses = checkpoint['train_losses']
            train_accuracies = checkpoint['train_accuracies']
            validation_losses = checkpoint['validation_losses']
            validation_accuracies = checkpoint['validation_accuracies']
            classes = checkpoint['classes']
            self.model.to(self.device)
            self.model.eval()

            # Plot and save an example image
            plot_image(self.validation_dataset, self.model, classes)


if __name__ == "__main__":
    resnet = Resnet()
    resnet.run(train_model=False)