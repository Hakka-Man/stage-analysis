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


def load_dataset(dataset_path):
    data_transforms = {
        'train': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
        'validation': transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]),
    }

    # Todo: Load dataset
    # dataset_path = '../../train_images'
    train_dataset = ImageFolder(root=dataset_path + "/train", transform=data_transforms['train'])
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)

    validation_dataset = ImageFolder(root=dataset_path + "/validation", transform=data_transforms['validation'])
    validation_loader = DataLoader(validation_dataset, batch_size=4, shuffle=True, num_workers=4)

    classes = ("1", "2", "3", "4")
    return train_dataset, train_loader, validation_dataset, validation_loader, classes


def train(model, train_loader, criterion, optimizer, device):
    train_loss = 0.0
    train_total = 0
    train_correct = 0

    model.train()

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

        _, predicted = torch.max(outputs, 1)
        train_total += labels.size(0)
        train_correct += (predicted == labels).sum().item()

    train_loss = train_loss / len(train_loader.dataset)
    train_accuracy = 100.0 * train_correct / train_total

    return model, train_loss, train_accuracy


def validation(model, validation_loader, criterion, device):
    validation_loss = 0.0
    validation_total = 0
    validation_correct = 0

    model.eval()

    with torch.no_grad():
        
        for inputs, labels in validation_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            validation_loss += loss.item() * inputs.size(0)

            _, predicted = torch.max(outputs, 1)
            validation_total += labels.size(0)
            validation_correct += (predicted == labels).sum().item()
    
    validation_loss = validation_loss / len(validation_loader.dataset)
    validation_accuracy = 100.0 * validation_correct / validation_total

    return validation_loss, validation_accuracy


def train_epochs(model, train_loader, validation_loader, criterion, optimizer, device, num_epochs, save_intervalidation=5):
    train_losses = []
    train_accuracies = []
    validation_losses = []
    validation_accuracies = []

    classes = ("1", "2", "3", "4")

    for epoch in range(num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        model, train_loss, train_accuracy = train(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_accuracy = validation(model, validation_loader, criterion, device)

        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        validation_losses.append(validation_loss)
        validation_accuracies.append(validation_accuracy)

        print(f'Train Loss: {train_loss:.4f} - Train Accuracy: {train_accuracy:.2f}%')
        print(f'validation Loss: {validation_loss:.4f} - validation Accuracy: {validation_accuracy:.2f}%')
        print()

        if (epoch + 1) % save_intervalidation == 0:
          # Save the model and variables
        #   torch.save(model.state_dict(), f'resnet50_cifar10_{epoch+1}.pth')
          checkpoint = {
              'epoch': epoch + 1,
              'train_losses': train_losses,
              'train_accuracies': train_accuracies,
              'validation_losses': validation_losses,
              'validation_accuracies': validation_accuracies,
              'classes': classes,
          }
          # torch.save(checkpoint, f'resnet50_stage_variables_{epoch+1}.pth')

    return model, train_losses, train_accuracies, validation_losses, validation_accuracies

def plot_loss(train_losses, validation_losses):
    plt.figure()
    plt.plot(range(len(train_losses)), train_losses, label='Training Loss')
    plt.plot(range(len(validation_losses)), validation_losses, label='validationidation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss_plot.png')
    plt.show()


def plot_accuracy(train_accuracies, validation_accuracies):
    plt.figure()
    plt.plot(range(len(train_accuracies)), train_accuracies, label='Training Accuracy')
    plt.plot(range(len(validation_accuracies)), validation_accuracies, label='validationidation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('accuracy_plot.png')
    plt.show()


def plot_image(dataset, model, classes):
    idx = random.randint(0, len(dataset))
    label = dataset[idx][1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = dataset[idx][0].unsqueeze(0).to(device)  # Move the input image tensor to the GPU
    model.eval()
    #model.to(device)  # Move the model to the GPU
    output = model(img)
    _, predicted = torch.max(output.data, 1)
    # Convert the image and show it
    img = img.squeeze().permute(1, 2, 0).cpu()  # Move the image tensor back to the CPU and adjust dimensions
    plt.imshow(img)
    plt.axis('off')
    plt.title(f'Predicted: {classes[predicted]}, True: {classes[label]}')
    plt.savefig('predicted_image.png')
    plt.show()
    print("Predicted label: ", classes[predicted[0].item()])
    print("Actual label: ", classes[label])