import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import random
import torch.optim as optim
import argparse
import warnings
import gc

# Use GPU if available, else use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

# Define the neural network model (DNN)
class DNN_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN_Net, self).__init__()
        # Layers of the network with batch normalization and dropout
        self.fc4 = nn.Linear(64, 59)
        self.bn4 = nn.BatchNorm1d(59)
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output_layer = nn.Linear(32, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x, concat=False): 
        # Forward pass with ReLU activations and dropout for regularization
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        output = F.softmax(self.output_layer(x))
        return output, x

# Function to set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Function to shuffle two arrays in unison
def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

# Function to normalize the input data (train and test data)
def normalize_data(train_x, test_x):
    for j in range(train_x.shape[2]):
        mean = np.mean(train_x[:, :, j])
        std = np.std(train_x[:, :, j])
        test_x[:, :, j] = (test_x[:, :, j] - mean) / std
        train_x[:, :, j] = (train_x[:, :, j] - mean) / std
    return train_x, test_x

# Function to train and evaluate the model
def train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, minority_class, majority_class, num_epochs=100, clip=1):
    best_test_accuracy = 0
    best_f1 = 0
    best_conf_matrix = None
    EPOCH = 0
    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []
    test_f1_list = []
    max_accuracy_after_20_epochs = 0
    max_f1_after_20_epochs = 0
    last_5_accuracies = []
    last_5_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total_loss = 0.0

        # Training loop
        for i, batch in enumerate(train_loader):
            inputs, targets = batch[0].cuda(), batch[1].cuda().view(-1).to(torch.float32)
            optimizer.zero_grad()  # Reset gradients
            output, pre_meta_inputs = model(inputs)  # Forward pass
            targets = targets.long()
            loss = criterion(output, targets)  # Calculate loss
            loss.backward()  # Backpropagate gradients
            nn.utils.clip_grad_norm_(model.parameters(), clip)  # Gradient clipping
            optimizer.step()  # Update model parameters
            pred = output.argmax(dim=1).cuda()  # Get predicted labels
            correct += int((pred == targets).sum())
            total_loss += loss.item()

        # Calculate training accuracy and loss
        train_accuracy = correct / len(train_loader.dataset)
        average_loss = total_loss / len(train_loader)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(average_loss)

        model.eval()
        correct = 0
        all_true_labels = []
        all_predicted_labels = []

        # Evaluation loop
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                test_inputs, test_targets = batch[0].cuda(), batch[1].cuda().view(-1).long()
                output, _ = model(test_inputs)  # Forward pass on test data
                test_targets = test_targets.cpu().numpy()
                predicted = output.argmax(dim=1)
                all_true_labels.extend(test_targets)
                all_predicted_labels.extend(predicted.cpu().numpy())
                correct += np.sum(predicted.cpu().numpy() == test_targets)

        # Calculate evaluation metrics
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
        test_accuracy_list.append(accuracy)
        test_f1_list.append(f1)

        # Track the best accuracy and F1 score
        if epoch >= 20:
            if accuracy > max_accuracy_after_20_epochs:
                max_accuracy_after_20_epochs = accuracy
                max_f1_after_20_epochs = f1

        if accuracy > best_test_accuracy:
            EPOCH = epoch
            best_test_accuracy = accuracy
            best_f1 = f1
            best_conf_matrix = conf_matrix

        # Print progress every 5 epochs
        if epoch % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Test Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {f1:.4f}')
        
        # Clear GPU memory and garbage collection
        torch.cuda.empty_cache()
        gc.collect()

    # After training, calculate and print results
    last_5_accuracies = test_accuracy_list[-5:]
    last_5_f1_scores = test_f1_list[-5:]
    mean_last_5_accuracy = np.mean(last_5_accuracies)
    mean_last_5_f1 = np.mean(last_5_f1_scores)

    print(f'EPOCH: {EPOCH}')
    print(f'Best Test Accuracy: {best_test_accuracy:.4f}')
    print(f'Best F1 Score: {best_f1:.4f}')
    print('Best Confusion Matrix:')
    print(best_conf_matrix)
    print(f'Max Accuracy After 20 Epochs: {max_accuracy_after_20_epochs:.4f}')
    print(f'Max F1 Score After 20 Epochs: {max_f1_after_20_epochs:.4f}')
    print(f'Mean Accuracy of Last 5 Epochs: {mean_last_5_accuracy:.4f}')
    print(f'Mean F1 Score of Last 5 Epochs: {mean_last_5_f1:.4f}')

    # Plot the results for visual analysis
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.plot(range(num_epochs), train_loss_list, label='Train Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()

    plt.subplot(1, 3, 2)
    plt.plot(range(num_epochs), train_accuracy_list, label='Train Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy')
    plt.legend()

    plt.subplot(1, 3, 3)
    plt.plot(range(num_epochs), test_accuracy_list, label='Test Accuracy')
    plt.savefig(f"{path}/{args.dataset}/wometa_parameters.jpg")

# Function to modify labels for a specific dataset
def modify_labels(y_train, y_test):
    y_train[y_train == 2] = 1  # Change label 2 to 1
    y_test[y_test == 2] = 1    # Change label 2 to 1
    return y_train, y_test

# Function to get the minority and majority class labels
def get_minority_majority_classes(targets):
    unique_classes, counts = np.unique(targets, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    return minority_class, majority_class

# Main execution block
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and process dataset") 
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (reRLDD or reDROZY)") 
    parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    set_seed(args.seed)  # Set the random seed for reproducibility

    # Load the dataset
    path='./fine_grained_feature'
    x_train = np.load(f"{path}/{args.dataset}/train.npy")
    x_test = np.load(f"{path}/{args.dataset}/test.npy")
    y_train = np.load(f"{path}/{args.dataset}/train_label.npy")
    y_test = np.load(f"{path}/{args.dataset}/test_label.npy")

    # Reshape data
    x_train = x_train.reshape(-1, 1, 59)
    x_test = x_test.reshape(-1, 1, 59)

    # Modify labels and shuffle the training data
    y_train, y_test = modify_labels(y_train, y_test)
    batch_size = 32
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_train, x_test = normalize_data(x_train, x_test)

    # Flatten the data for input to the model
    x_train = x_train.reshape(-1, 59)
    x_test = x_test.reshape(-1, 59)

    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_test.shape", y_test.shape)

    # Create DataLoader for training and testing
    train_loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size, shuffle=False, drop_last=True)

    # Define model, loss, optimizer
    num_epochs = args.epoch
    input_size = x_train.shape[1]
    output_size = 2
    criterion = nn.CrossEntropyLoss()
    model = DNN_Net(input_size, output_size).cuda()
    learning_rate = 0.001
    weight_decay = 1e-3  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Get the minority and majority class
    minority_class, majority_class = get_minority_majority_classes(y_train)

    # Train and evaluate the model
    train_and_evaluate(model, train_loader, test_loader, criterion, optimizer, minority_class, majority_class, clip=1, num_epochs=num_epochs)
