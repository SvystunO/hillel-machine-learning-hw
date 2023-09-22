import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary
import matplotlib.pyplot as plt
import math
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import random


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)     # <- (3*3 + 1)*32 = 320   (3*3 + 1)*64 640
        self.conv2 = nn.Conv2d(64, 128, 3, 1)    # <- (3*3)*32*64 + 64 = 18496 (3*3)*64*128 + 128 73856
        self.bn1 = nn.BatchNorm2d(64)  # Batch normalization after conv1
        self.bn2 = nn.BatchNorm2d(128)  # Batch normalization after conv2
        self.dropout1 = nn.Dropout(0.25)        # <- 0
        self.dropout2 = nn.Dropout(0.5)         # <- 0
        self.fc1 = nn.Linear(2048, 128)         # <- 9216*128 + 128 = 1179776  2048*128 + 128 = 262272
        self.fc2 = nn.Linear(128, 10)           # <- 128*10 + 10 = 1290 128*10 = 1290

    def forward(self, x):           # [128,  1, 28, 28] [128, 1, 28, 28]
        x = self.conv1(x)           # [128, 32, 26, 26] [128, 64, 28, 28]
       # x = self.bn1(x)
        x = F.relu(x)               # [128, 32, 26, 26] [128, 64, 26, 26]
        x = self.conv2(x)           # [128, 64, 24, 24] [128, 128, 24, 24]
      #  x = self.bn2(x)
        x = F.relu(x)               # [128, 64, 24, 24] [128, 128, 24, 24]
        x = F.max_pool2d(x, 6)      # [128, 128, 12, 12] [128, 128, 4, 4]
        x = self.dropout1(x)        # [128, 64, 12, 12] [128, 128, 4, 4]
        x = torch.flatten(x, 1)     # [128, 64x12x12 = 9216] [128, 128x4x4 = 2048]
        x = self.fc1(x)             # [128, 128]
        x = F.relu(x)               # [128, 128]
        x = self.dropout2(x)        # [128, 128]
        output = self.fc2(x)        # [128, 10]
        return output


def train(model, device, train_loader, optimizer, epoch, loss_fn):
    model.train()
    batch_in_epoch_size = 0
    loss_epoch = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        batch_in_epoch_size += len(data)
        loss = loss_fn(output, target)
        loss.backward()
        loss_epoch += loss.item();
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
       # print(loss_epoch, batch_in_epoch_size)
    los_per_train = math.log(loss_epoch / batch_in_epoch_size)
    return los_per_train

def test(model, device, test_loader, loss_fn):
    model.eval()
    test_loss = 0
    correct = 0
    #labels of classes
    class_labels = [0,1,2,3,4,5,6,7,8,9]
    num_classes = len(class_labels)
    #initialize confusion matrix
    conf_matrix = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += loss_fn(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            conf_matrix += confusion_matrix(target.cpu(), pred.cpu(), labels=list(range(num_classes)))

    test_loss /= len(test_loader.dataset)
    class_accuracies = conf_matrix.diag() / conf_matrix.sum(1)

    print(class_accuracies)
    print(conf_matrix)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return test_loss

def main(epochs=3):

    torch.manual_seed(1337)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    overfit = True

    train_kwargs = {'batch_size': 128}
    test_kwargs = {'batch_size': 128}

    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    train_dataset = datasets.MNIST('./mnsit-dataset', train=True, download=True,
                              transform=transform)
    if overfit:
        samples_per_class = 200

        # Define the transformation
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])

        # Load the full MNIST training dataset
        full_train_dataset = train_dataset

        # Create a list to store the selected indices
        selected_indices = []

        # Iterate through each class and select samples
        for class_label in range(10):
            class_indices = [i for i, (_, label) in enumerate(full_train_dataset) if label == class_label]
            selected_indices.extend(random.sample(class_indices, samples_per_class))

        # Create a Subset of the training dataset with the selected indices
        train_dataset = torch.utils.data.Subset(full_train_dataset, selected_indices)

    print(len(train_dataset))

    test_dataset = datasets.MNIST('./mnsit-dataset', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(test_dataset, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=0.1)
    loss_fn = torch.nn.CrossEntropyLoss()
    train_loss_arr = []
    test_loss_arr = []
    scheduler = StepLR(optimizer, step_size=1, gamma=0.5)
    for epoch in range(1, epochs + 1):
        train_loss_arr.append(train(model, device, train_loader, optimizer, epoch, loss_fn))
        test_loss_arr.append(math.log(test(model, device, test_loader, loss_fn)))

        scheduler.step()


    torch.save(model.state_dict(), "mnist_cnn.pt")
    print(f"# Params: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    print('Model:')
    print(model)
    print('Summary report:')
    print(summary(model, input_size=(1, 28, 28)))

    return train_loss_arr, test_loss_arr


if __name__ == '__main__':

    epochs = 3

    train_loss_arr, test_loss_arr = main(epochs=epochs)

    num = [i for i in range(1, len(train_loss_arr) + 1)]
    fig, ax = plt.subplots(figsize=(8, 6))

    ln_train_loss = train_loss_arr
    ln_test_loss = test_loss_arr
    ax.plot(ln_train_loss, label="Train Loss function", color="green")
    #ax.plot(ln_test_loss, label="Test Loss function", color="red")

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

    plt.title("loss function")
    plt.savefig("loss_1arr_1_"+str(epochs)+".png")
    plt.close('all')