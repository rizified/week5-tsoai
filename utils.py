import matplotlib.pyplot as plt
from torchsummary import summary
from torchvision import datasets, transforms
import torch
from tqdm import tqdm

#This Utility class contains the methods which are required for data related tasks like transformations,splitting,viewing and also to plot metrics of model
class Utility:
    def train_transforms(self):
        # Train data transformations
        return transforms.Compose([
            transforms.RandomApply([transforms.CenterCrop(22), ], p=0.1),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15., 15.), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
            ])

    def test_transforms(self):
        # Test data transformations
        return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    def view_data_plot(self,batch_data,batch_label,items,rows,columns):
        fig = plt.figure()
        for i in range(items):
            plt.subplot(rows,columns,i+1)
            plt.tight_layout()
            plt.imshow(batch_data[i].squeeze(0), cmap='gray')
            plt.title(batch_label[i].item())
            plt.xticks([])
            plt.yticks([])

    def plot_metrics(self,train_losses, train_accuracy, test_losses, test_accuracy):
        # Code to plot the metrics
        fig, axs = plt.subplots(2,2,figsize=(15,10))
        axs[0, 0].plot(train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(train_accuracy)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(test_accuracy)
        axs[1, 1].set_title("Test Accuracy")

    def dataloaders_split(self,batch_size = 512,shuffle = True, num_workers = 2,pin_memory = True):

        train_data = datasets.MNIST('../data', train=True, download=True, transform=self.train_transforms())
        test_data = datasets.MNIST('../data', train=False, download=True, transform=self.test_transforms())
        kwargs = {'batch_size': batch_size, 'shuffle': shuffle, 'num_workers': num_workers, 'pin_memory': pin_memory}

        test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
        train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
        return train_loader,test_loader


#This Model Helper Utility class contains the methods which are required for Training,Testing and viewing summary of the model.
class ModelHelpers:
    def __init__(self,model,device,criterion):
        self.model = model.to(device)
        self.criterion = criterion
        self.device = device
    
    def GetCorrectPredCount(self,pPrediction, pLabels):
        return pPrediction.argmax(dim=1).eq(pLabels).sum().item()
    
    def train(self, train_loader, optimizer):
        self.model.train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            # Predict
            pred = self.model(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            train_loss+=loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += self.GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(desc= f'Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}')

        return 100*correct/processed,train_loss/len(train_loader)
    
    def test(self,test_loader):
        self.model.eval()

        test_loss = 0
        correct = 0
        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self.model(data)
                test_loss += self.criterion(output, target, reduction='sum').item()  # sum up batch loss

                correct += self.GetCorrectPredCount(output, target)


        test_loss /= len(test_loader.dataset)
        test_accuracy = 100. * correct / len(test_loader.dataset)
        return  test_accuracy, test_loss

        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))
    
    def view_summary(self,input_size):
        return summary(self.model, input_size=input_size)


