# MNIST - Neural Network Solution

This repository contains a neural network solution for MNIST dataset.The solution includes utility functions, a neural network model, and a main script for training and testing the model. 

![Responsice Mockup](https://static.javatpoint.com/tutorial/tensorflow/images/mnist-dataset-in-cnn.jpg)

## File Structure

The repository has the following file structure:

- `utils.py`: This file contains utility functions for data-related tasks such as data transformations, data visualization, and plotting model metrics.

- `model.py`: This file defines the neural network model as a subclass of `nn.Module`. The model consists of four convolutional layers and two fully connected layers and also has forward method which has maxpooling layers and activation functions.

- `S5.ipynb`: This file serves as the main script for training and testing the neural network model. It imports the necessary modules, including the `NNModel` class, `Utility` class, and the `ModelHelpers` class.

## Usage
To use the solution, follow these steps:

- Clone the repository:
    
    ```
    git clone https://github.com/rizified/week5-tsoai.git
    ```
- Import the required modules and classes:
    ```
    import torch
    import torch.optim as optim
    from model import NNModel, get_loss_function
    from utils import ModelHelpers, Utility
    ```

- Create an instance of the `NNModel` class, `ModelHelpers` class, and `Utility` class:

    ```
    model_obj = NNModel(bias_value=True)
    model_helpers = ModelHelpers(model=model_obj, device=device, criterion=get_loss_function())
    utility_obj = Utility()
    ```

- Split the dataset into train and test dataloaders using the `dataloaders_split()` method of the `Utility` class:

    ```
    train_dataloader, test_dataloader = utility_obj.dataloaders_split(batch_size=512, shuffle=True, num_workers=10, pin_memory=True)
    ```
- Visualize the data using the `view_data_plot()` method of the `Utility` class:

    ```
    batch_data, batch_label = next(iter(train_dataloader))
    utility_obj.view_data_plot(batch_data, batch_label, 12, 3, 4)
    ```
- View the summary of the model using the `view_summary()` method of the `ModelHelpers` class:

    ```
    model_helpers.view_summary(input_size=(1, 28, 28))
    ```

- Initialize the loss function, optimizer, and scheduler:

    ```
    optimizer = optim.SGD(model_obj.parameters(), lr=0.01, momentum=0.9)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=15, gamma=0.1, verbose=True)
    ````

    _Make sure to adjust the parameters and paths according to your specific requirements._


- Train and test the model for a specified number of epochs using the train() and test() methods of the ModelHelpers class:

    ```
    num_epochs = 20
    train_losses = []
    test_losses = []
    train_acc = []
    test_acc = []

    for epoch in range(1, num_epochs+1):
        train_accuracy, train_loss_value = model_helpers.train(train_dataloader, optimizer)
        train_acc.append(train_accuracy)
        train_losses.append(train_loss_value)
        test_accuracy, test_loss_value = model_helpers.test(test_dataloader)
        test_acc.append(test_accuracy)
        test_losses.append(test_loss_value)
        scheduler.step()
    ```

- Plot the training and test metrics using the `plot_metrics()` method of the `Utility `class:

    ```
   utility_obj.plot_metrics(train_losses, train_acc, test_losses, test_acc)
    ```

- All this are present in S5.ipynb file.

### Additional Files/References
`S4-Solution.ipynb` was reffered during building the solution.
