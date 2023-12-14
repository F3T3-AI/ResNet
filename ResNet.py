import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import argparse
import pandas as pd
import os
import numpy as np
from torch.utils.data import DataLoader
from torchvision.models.mobilenetv3 import mobilenet_v3_large
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from dataset import MyDataset
from tqdm import tqdm
from torchvision.models import resnet101



class Classifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []

    def train(self, train_loader, val_loader, epochs, optimizer, criterion, start_epoch=0):
        best_val_acc = 0.0
        print("Train....")

        for epoch in range(start_epoch, epochs):
            train_loss = 0.0
            val_loss = 0.0
            train_acc = 0.0
            val_acc = 0.0

            self.model.train()
            train_loader_iter = tqdm(train_loader, desc=(f"Epoch: {epoch + 1}/{epochs}"), leave=False)

            for i, (data, target) in enumerate(train_loader_iter):
                data, target = data.float().to(self.device), target.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(data)
                loss = criterion(outputs, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

                _, pred = torch.max(outputs, 1)
                train_acc += (pred == target).sum().item()

                train_loader_iter.set_postfix({"Loss": loss.item()})

            train_loss /= len(train_loader)
            train_acc = train_acc / len(train_loader.dataset)

            self.model.eval()
            with torch.no_grad():
                for data, target in val_loader:
                    data, target = data.float().to(self.device), target.to(self.device)
                    output = self.model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    val_acc += pred.eq(target.view_as(pred)).sum().item()
                    val_loss += criterion(output, target).item()

            val_loss /= len(val_loader)
            val_acc = val_acc / len(val_loader.dataset)

            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)

            if val_acc > best_val_acc:
                torch.save(self.model.state_dict(), "./ex01_0717_resnet50_best.pt")
                best_val_acc = val_acc

            print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss:.4f}, "
                  f"Val loss: {val_loss:.4f}, Train ACC: {train_acc:.4f}, Val ACC: {val_acc:.4f}")

            # Save the model state and optimizer state after each epoch
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_losses': self.train_losses,
                'train_accs': self.train_accs,
                'val_losses': self.val_losses,
                'val_accs': self.val_accs
            }, f'.//saves//{epoch+1}_ex01_0717_resnet50_checkpoint.pt')

        torch.save(self.model.state_dict(), "./ex01_0717_resnet50_last.pt")

        #print(f"Epoch [{epoch + 1}/{epochs}], Train loss: {train_loss:.4f}, Train ACC: {train_acc:.4f}")

        # After training is complete:
        self.save_results_to_csv()
        self.plot_loss()
        self.plot_accuracy()

    def save_results_to_csv(self):
        df = pd.DataFrame({
            'Train Loss': self.train_losses,
            'Train Accuracy': self.train_accs,
            'Validation Loss': self.val_losses,
            'Validation Accuracy': self.val_accs
        })
        df.to_csv('train_val_results_ex01.csv', index=False)

    def plot_loss(self):
        plt.figure()
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig('ex01_loss_plot.png')

    def plot_accuracy(self):
        plt.figure()
        plt.plot(self.train_accs, label='Train Accuracy')
        plt.plot(self.val_accs, label='Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.savefig('ex01_accuracy_plot.png')

    def run(self, args):
        self.model = resnet101(pretrained=True)
        self.model.fc = nn.Linear(2048, 20)
        self.model.to(self.device)

        train_transforms = transforms.Compose([
            transforms.Resize(250),
            transforms.RandomAffine(degrees=15, translate=(0.05, 0.05), scale=(0.95, 1.05)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=5),
            transforms.AugMix(),
            transforms.RandomVerticalFlip(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        val_transforms = transforms.Compose([
            transforms.Resize(250),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        train_dataset = MyDataset(path = args.train_dir, transforms=train_transforms)
        val_dataset = MyDataset(path = args.val_dir, transforms=val_transforms)

        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

        epochs = args.epochs
        criterion = CrossEntropyLoss().to(self.device)
        optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)

        start_epoch = 0
        if args.resume_training:
            checkpoint = torch.load(args.checkpoint_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.train_losses = checkpoint['train_losses']
            self.train_accs = checkpoint['train_accs']
            self.val_losses = checkpoint['val_losses']
            self.val_accs = checkpoint['val_accs']
            start_epoch = checkpoint['epoch']

        self.train(train_loader, val_loader, epochs, optimizer, criterion, start_epoch=start_epoch)

#    def epoch_class_accuracy(self, dataloader):
#         self.model.eval()
#         class_correct = np.zeros(self.num_classes)
#         class_total = np.zeros(self.num_classes)

#         with torch.no_grad():
#             for data, target in dataloader:
#                 data, target = data.float().to(self.device), target.to(self.device)
#                 outputs = self.model(data)
#                 _, preds = torch.max(outputs, 1)

#                 corrects = preds.eq(target.view_as(preds)).cpu().numpy()

#                 for i in range(self.num_classes):
#                     class_correct[i] += corrects[target == i].sum()
#                     class_total[i] += (target == i).sum()

#         class_accuracies = class_correct / class_total
#         for i in range(self.num_classes):
#             print(f'Class {i} Accuracy: {class_accuracies[i]:.4f}')

def most_recent_epoch(directory):
    file_list = []
    biggest = 0
    for filename in os.listdir(directory):
        if os.path.isfile(os.path.join(directory, filename)):
            file_list.append(filename)
    for epoch in file_list:
        epoch_name = int(epoch.split('_')[0])
        if epoch_name > biggest:
            biggest = epoch_name
            print(f"biggest changed : {biggest}")
    file = str(biggest)+"_ex01_0717_resnet50_checkpoint.pt"
    full_path = os.path.join(directory, file)
    return full_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    file_path = most_recent_epoch("Model save path")
    parser.add_argument('--train_dir', type=str, default="training path",
                        help='directory path to the training dataset')
    parser.add_argument('--val_dir', type=str, default="val path",
                        help='directory path to the validation dataset')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=164,
                        help='batch size for training and validation')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate for optimizer')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='weight decay for optimizer')
    parser.add_argument('--resume_training', action='store_true', default=True,
                        help='resume training from the last checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default= file_path,
                        help='path to the checkpoint file')
    args = parser.parse_args()

    classifier = Classifier()
    classifier.run(args)