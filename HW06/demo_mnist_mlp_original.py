"""
Dataset references:
http://pjreddie.com/media/files/mnist_train.csv
http://pjreddie.com/media/files/mnist_test.csv
"""


from pathlib import Path
from typing import Tuple
import torch
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import numpy as np
import math


torch.manual_seed(1337)


class MnistMlp(torch.nn.Module):
    
    def __init__(self, inputnodes: int, hiddennodes: int, outputnodes: int) -> None:
        super().__init__()

        # number of nodes (neurons) in input, hidden, and output layer
        self.wih = torch.nn.Linear(in_features=inputnodes, out_features=hiddennodes)
        self.who = torch.nn.Linear(in_features=hiddennodes, out_features=outputnodes)
        self.activation = torch.nn.ELU()
        self.dropout = torch.nn.Dropout(p=dropout_prob)
        self.bn1 = torch.nn.BatchNorm1d(hiddennodes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.wih(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.dropout(out)
        out = self.who(out)
        return out


class MnistDataset(Dataset):
    
    def __init__(self, filepath: Path) -> None:
        super().__init__()

        self.data_list = None
        with open(filepath, "r") as f:
            self.data_list = f.readlines()

        # conver string data to torch Tensor data type
        self.features = []
        self.targets = []
        for record in self.data_list:
            all_values = record.split(",")
            features = np.asfarray(all_values[1:])
            target = int(all_values[0])
            self.features.append(features)
            self.targets.append(target)

        self.features = torch.tensor(np.array(self.features), dtype=torch.float) / 255.0
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.long)
        # print(self.features.shape)
        # print(self.targets.shape)
        # print(self.features.max(), self.features.min())

    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[index], self.targets[index]


if __name__ == "__main__":
    # Device for training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # NN architecture:
    # number of input, hidden and output nodes
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10
    # in case dropout is used drop rate
    dropout_prob = 0.2
    # learning rate is 0.1
    learning_rate = 0.1
    # batch size
    batch_size = 10
    # number of epochs
    epochs = 20
    # other condition used forplot title
    other_condition = 'ELU with BatchNorm1d'

    # Load mnist training and testing data CSV file into a datasets
    train_dataset = MnistDataset(filepath="./mnist_train.csv")
    test_dataset = MnistDataset(filepath="./mnist_test.csv")

    # Make data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    # Define NN
    model = MnistMlp(inputnodes=input_nodes, 
                     hiddennodes=hidden_nodes, 
                     outputnodes=output_nodes)
    # Number of parameters in the model
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    model = model.to(device=device)
    
    # Define Loss
    criterion = torch.nn.CrossEntropyLoss()

    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    ##### Training! #####
    model.train()

    train_loss_arr = []
    test_loss_arr = []
    train_accuracy_arr = []
    test_accuracy_arr = []


    for epoch in range(epochs):
        batch_in_epoch_size = 0
        epoch_correct_train = 0
        loss_epoch = 0
        for batch_idx, (features, target) in enumerate(train_loader):
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(features)
            batch_in_epoch_size += len(features)
            pred = output.argmax(dim=1, keepdim=True)
            epoch_correct_train += pred.eq(target.view_as(pred)).sum().item()
            loss = criterion(output, target)
            loss_epoch += loss.item();
            pred = output.argmax(dim=1, keepdim=True)

            loss.backward()
            optimizer.step()

        # adding train result data
        train_loss_arr.append(math.log(loss_epoch / batch_in_epoch_size))
        train_accuracy_arr.append(epoch_correct_train / batch_in_epoch_size * 100)
        ##### Testing! #####
        model.eval()
        test_loss = 0
        correct = 0
        with torch.inference_mode():
            for features, target in test_loader:
                features, target = features.to(device), target.to(device)
                output = model(features)
                #    print('output: ' + str(output) + ' target: ' + str(target))
                test_loss += criterion(output, target).item()  # sum up batch loss
                #    print('loss:' + str(test_loss) )
                pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                #    print('pred: ' + str(pred))
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(test_loader.dataset)
        test_correct = correct / len(test_loader.dataset) * 100
        test_loss_arr.append(math.log(test_loss))
        test_accuracy_arr.append(test_correct)
        ##### Switch back to train #####
        model.train()


    ##### Testing! #####
    model.eval()
    test_loss = 0
    correct = 0
    with torch.inference_mode():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            output = model(features)
        #    print('output: ' + str(output) + ' target: ' + str(target))
            test_loss += criterion(output, target).item()  # sum up batch loss
        #    print('loss:' + str(test_loss) )
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #    print('pred: ' + str(pred))
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy_fin = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    print(train_loss_arr)
    print(train_accuracy_arr)

    print(test_loss_arr)
    print(test_accuracy_arr)




    # Generate plot with Train loss and Test loss function
    num = [i for i in range(1, len(train_loss_arr) + 1)]
    fig, ax = plt.subplots(figsize=(8, 6))

    ln_train_loss = train_loss_arr
    ln_test_loss = test_loss_arr
    ax.plot(ln_train_loss, label="Train Loss function", color="green")
    ax.plot(ln_test_loss, label="Test Loss function", color="red")

    if min(ln_train_loss) <= min(ln_test_loss):
        miny = min(ln_train_loss) - 2
    else:
        miny = min(ln_test_loss) - 2

    if max(ln_train_loss) >= max(ln_test_loss):
        maxy = max(ln_train_loss) + 2
    else:
        maxy = max(ln_test_loss) + 2

    plt.ylim(miny, maxy)

    ax.legend(loc="best")

    plt.title("" + other_condition + " dropout: " + str(dropout_prob) + ", batch_size: " + str(batch_size) + ", epochs:" + str(epochs) + ", accuracy: " + str(accuracy_fin))
    plt.savefig("HW/loss_arr_true_" + str(learning_rate) + "_" + str(batch_size) + "_" + str(epochs) + "_" + str(
        dropout_prob) + "_" + str(len(other_condition)) + ".png")
    plt.close('all')



    # accuracy plot  generation

    num = [i for i in range(1, len(train_accuracy_arr) + 1)]
    fig, ax = plt.subplots(figsize=(8, 6))


    ax.plot(train_accuracy_arr, label="Train Accuracy", color="green")
    ax.plot(test_accuracy_arr, label="Test Accuracy", color="red")

    if min(train_accuracy_arr) <= min(test_accuracy_arr):
        miny = min(train_accuracy_arr) - 2
    else:
        miny = min(test_accuracy_arr) - 2

    if max(train_accuracy_arr) >= max(test_accuracy_arr):
        maxy = max(train_accuracy_arr) + 2
    else:
        maxy = max(test_accuracy_arr) + 2

    plt.ylim(miny, maxy)

    ax.legend(loc="best")

    plt.title("Accuracy" + other_condition + " dropout: " + str(dropout_prob)+ ", batch_size: " + str(batch_size) + ", epochs:" + str(epochs) + ", accuracy: " + str(accuracy_fin))
    plt.savefig("HW/accuracy_" + str(learning_rate) + "_" + str(batch_size) + "_" + str(epochs) + "_" + str(
        dropout_prob) + "_" + str(len(other_condition)) + ".png")
    plt.close('all')

    ##### Save Model! #####
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "mnist_001.pth")
