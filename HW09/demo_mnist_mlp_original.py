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
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay




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
    batch_size = 50
    # number of epochs
    epochs = 5
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
    #labels of classes
    class_labels = [0,1,2,3,4,5,6,7,8,9]
    num_classes = len(class_labels)
    #initialize confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    all_pred_labels = []
    all_true_labels = []
    with torch.inference_mode():
        for features, target in test_loader:
            features, target = features.to(device), target.to(device)
            output = model(features)
        #    print('output: ' + str(output) + ' target: ' + str(target))
            test_loss += criterion(output, target).item()  # sum up batch loss
        #    print('loss:' + str(test_loss) )
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        #    print('pred: ' + str(pred))
            all_pred_labels.extend(pred.cpu().numpy())
            all_true_labels.extend(target.cpu().numpy())
            correct += pred.eq(target.view_as(pred)).sum().item()
            #update confusion matrix by prediction
            # .cpu() - move tensors to cpu where matrix is operated
            conf_matrix += confusion_matrix(target.cpu(), pred.cpu(), labels=list(range(num_classes)))

    test_loss /= len(test_loader.dataset)
    accuracy_fin = 100. * correct / len(test_loader.dataset)
    class_accuracies = conf_matrix.diag() / conf_matrix.sum(1)

    print(class_accuracies)
    print(conf_matrix)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    # Calculate recall, precision, and F1-score for each class

    #conf_matrix.diag() get diagonal values and div on total count of actual instances (true positives + false negatives) horizontal (row)
    class_recall = conf_matrix.diag() / conf_matrix.sum(1)
    # conf_matrix.diag() get diagonal values and div on total count of actual instances (true positives + false positives) vertical (column)
    class_precision = conf_matrix.diag() / conf_matrix.sum(0)
    class_f1_score = 2 * (class_precision * class_recall) / (class_precision + class_recall)

    print("class_recall:")
    print(class_recall)
    print("class_precision:")
    print(class_precision)
    print("class_f1_score:")
    print(class_f1_score)


    # Calculate overall recall, precision, and F1-score
    overall_recall = conf_matrix.diag().sum() / conf_matrix.sum()
    overall_precision = conf_matrix.diag().sum() / conf_matrix.sum()
    overall_f1_score = 2 * (overall_precision * overall_recall) / (overall_precision + overall_recall)

    print("overall_recall:")
    print(overall_recall)
    print("overall_precision:")
    print(overall_precision)
    print("overall_f1_score:")
    print(overall_f1_score)

    # Convert the confusion matrix to a numpy array
    conf_matrix_np = conf_matrix.numpy()

    # Create a ConfusionMatrixDisplay
    disp = ConfusionMatrixDisplay(conf_matrix_np, display_labels=class_labels)

    # Display the confusion matrix plot
    plt.figure(figsize=(10, 8))
    disp.plot(cmap=plt.cm.Blues, values_format='d')
    plt.title('Confusion Matrix')
    #plt.show()
    plt.savefig('confusion_matrix.png')


    # Generate the classification report
    target_names = [f'Class {i}' for i in range(num_classes)]

    all_pred_labels = np.array(all_pred_labels)
    all_true_labels = np.array(all_true_labels)

    report = classification_report(all_true_labels, all_pred_labels)

    # Print the classification report
    print(report)

    ##### Save Model! #####
    # https://pytorch.org/tutorials/beginner/saving_loading_models.html
    torch.save(model.state_dict(), "mnist_001.pth")
    exit()