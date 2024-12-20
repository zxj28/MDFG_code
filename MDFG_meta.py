import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import random
from torch.utils.data import DataLoader, Dataset
import warnings
import gc
import argparse
import time

# Define the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

class DNN_Net(nn.Module):
    def __init__(self, input_size, output_size):
        super(DNN_Net, self).__init__()
        self.fc4 = nn.Linear(64, 59)
        self.bn4 = nn.BatchNorm1d(59)
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.output_layer = nn.Linear(32, output_size)
        self.output_layer1 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x,concat=False):
        """
        Forward pass of the network. The function takes input `x` and processes it through the layers.
        If `concat` is True, it adds extra operations at the end.
        """
        if concat:
            x = torch.relu(self.bn4(self.fc4(x)))  
        x = torch.relu(self.bn1(self.fc1(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn2(self.fc2(x)))
        x = self.dropout(x)
        x = torch.relu(self.bn3(self.fc3(x)))
        x = self.dropout(x)
        output = F.softmax(self.output_layer(x))

        if concat:
            x= self.output_layer1(x)
            x= self.Sigmoid(x)
            return x
        return output,x

def construct_meta_batch(pre_meta_inputs, targets, minority_class, majority_class, validation_split=0.2):
    while True:
        num_samples = len(pre_meta_inputs)
        indices = list(range(num_samples))
        split = int(np.floor(validation_split * num_samples))
        np.random.shuffle(indices)  # Shuffle data for randomness
        train_idx, val_idx = indices[split:], indices[:split]  # Split into training and validation sets

        minority_val_indices = np.where(targets[val_idx] == minority_class)[0]
        majority_train_indices = np.where(targets[train_idx] == majority_class)[0]

        if len(minority_val_indices) == 0 or len(majority_train_indices) == 0:
            continue  # Ensure both minority and majority classes are present in the split

        A_idx = random.choice(minority_val_indices)  # Randomly pick a minority sample from validation
        A = pre_meta_inputs[val_idx][A_idx]
        meta_inputs = []
        meta_targets = []

        # Create the positive pair (A, A) with label 1
        meta_inputs.append(torch.cat((torch.tensor(A).float().cuda(), torch.tensor(A).float().cuda())))
        meta_targets.append(1)

        # Create negative pairs with samples from the majority class (B)
        B_indices = random.sample(list(majority_train_indices), 4)  # Pick 4 majority samples
        for idx in B_indices:
            B = pre_meta_inputs[train_idx][idx]
            meta_inputs.append(torch.cat((torch.tensor(A).float().cuda(), torch.tensor(B).float().cuda())))
            meta_targets.append(0)

        meta_inputs = torch.stack(meta_inputs)  # Stack inputs into a batch
        meta_targets = torch.tensor(meta_targets).float().cuda()  # Convert targets to tensor

        return meta_inputs, meta_targets  # Return the meta batch

# Meta-learning loss function (binary cross-entropy)
def meta_learning_loss(pairwise_scores, targets, inputs):
    total_loss = 0
    for score, y_j in zip(pairwise_scores, targets):
        total_loss += y_j * torch.log(score) + (1 - y_j) * torch.log(1 - score)  # Cross-entropy computation
    total_loss = -(1 / len(inputs)) * total_loss  # Average loss over batch
    return total_loss

# Set random seed for reproducibility
def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

# Custom dataset class for loading data
class MyDataSet(Dataset):
    def __init__(self, x_train, x_label):
        self.x_train = x_train  # Features
        self.x_label = x_label  # Labels

    def __len__(self):
        return self.x_train.shape[0]  # Number of samples

    def __getitem__(self, idx):
        return self.x_train[idx], self.x_label[idx]  # Get sample and label

# Get the current random seed (from numpy state)
def get_random_seed():
    state = np.random.get_state()
    seed = state[1][0]  # Extract seed value
    return seed

# Shuffle two arrays in unison (same permutation for both)
def unison_shuffled_copies(a, b):
    print(get_random_seed())  # Print the current seed
    p = np.random.permutation(len(a))  # Generate a random permutation
    return a[p], b[p]  # Shuffle both arrays

# Normalize data (zero mean, unit variance per feature)
def normalize_data(train_x, test_x):
    for j in range(train_x.shape[2]):  # Iterate over features (axis 2)
        mean = np.mean(train_x[:, :, j])  # Mean of feature j in train data
        std = np.std(train_x[:, :, j])  # Standard deviation of feature j in train data
        test_x[:, :, j] = (test_x[:, :, j] - mean) / std  # Normalize test data
        train_x[:, :, j] = (train_x[:, :, j] - mean) / std  # Normalize train data
    return train_x, test_x

# Modify labels (change class '2' to class '1')
def modify_labels(y_train, y_test):
    y_train[y_train == 2] = 1  # Change class 2 to class 1 in train set
    y_test[y_test == 2] = 1  # Change class 2 to class 1 in test set
    return y_train, y_test  # Return modified labels


def train_and_evaluate(model,meta_model,train_loader, test_loader, criterion, optimizer, minority_class, majority_class, num_epochs=100, clip=1):
    best_test_accuracy = 0
    best_f1 = 0
    best_conf_matrix = None
    EPOCH = 0
    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []
    test_f1_list = []
    start_time = time.time()
    max_accuracy_after_20_epochs = 0
    max_f1_after_20_epochs = 0
    last_5_accuracies = []
    last_5_f1_scores = []

    for epoch in range(num_epochs):
        model.train()
        meta_model.train()
        correct = 0
        total_loss = 0.0
        
        for i, batch in enumerate(train_loader):
            inputs, targets = batch[0].cuda(), batch[1].cuda().view(-1).to(torch.float32)
            optimizer.zero_grad()
            output, pre_meta_inputs = model(inputs)
            targets = targets.long()
            loss = criterion(output, targets)

            meta_inputs, meta_targets = construct_meta_batch(pre_meta_inputs, targets.cpu().numpy(),minority_class, majority_class)
            meta_inputs = meta_inputs.cpu().numpy()
            meta_targets = meta_targets.cpu().numpy()
            meta_inputs = torch.from_numpy(meta_inputs).float().cuda()
            meta_targets = torch.tensor(meta_targets).long()

            r_aj=model(meta_inputs,concat=True)
            meta_loss = meta_learning_loss(r_aj,meta_targets,inputs)

            loss = loss + meta_loss
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            pred = output.argmax(dim=1).cuda()
            correct += int((pred == targets).sum())
            total_loss += loss.item()

        train_accuracy = correct / len(train_loader.dataset)
        average_loss = total_loss / len(train_loader)
        train_accuracy_list.append(train_accuracy)
        train_loss_list.append(average_loss)

        model.eval()
        correct = 0
        all_true_labels = []
        all_predicted_labels = []

        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                test_inputs, test_targets = batch[0].cuda(), batch[1].cuda().view(-1).long()
                output, _ = model(test_inputs)
                output.cuda()
                test_targets = test_targets.cpu().numpy()
                predicted = output.argmax(dim=1)
                all_true_labels.extend(test_targets)
                all_predicted_labels.extend(predicted.cpu().numpy())
                correct += np.sum(predicted.cpu().numpy() == test_targets)

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
        test_accuracy_list.append(accuracy)
        test_f1_list.append(f1)

        if epoch >= 20:
            if accuracy > max_accuracy_after_20_epochs:
                max_accuracy_after_20_epochs = accuracy
                max_f1_after_20_epochs = f1

        if accuracy > best_test_accuracy:
            EPOCH = epoch
            best_test_accuracy = accuracy
            best_f1 = f1
            best_conf_matrix = conf_matrix

        if epoch % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Test Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {f1:.4f}')
        torch.cuda.empty_cache()
        gc.collect()

    end_time = time.time()
    total_duration = end_time - start_time
    print(f'Total training time: {total_duration:.2f} seconds')

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
    plt.savefig("11.jpg")

def modify_labels(y_train, y_test):
    y_train[y_train == 2] = 1
    y_test[y_test == 2] = 1
    return y_train, y_test

def get_minority_majority_classes(targets):
    unique_classes, counts = np.unique(targets, return_counts=True)
    minority_class = unique_classes[np.argmin(counts)]
    majority_class = unique_classes[np.argmax(counts)]
    return minority_class, majority_class
def get_numpy_seed():
    state = np.random.get_state()
    seed = state[1][0]
    return seed
def get_random_seed():
    return random.getstate()[1][0]  

if __name__ == "__main__":
     # Argument parsing for dataset and training configurations
    parser = argparse.ArgumentParser(description="Load and process dataset") 
    parser.add_argument("--dataset", type=str, required=True, help="Dataset to use (reRLDD or reDROZY)") 
    parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)  # Set the seed for reproducibility 

    # Load the dataset
    path = './fine_grained_feature'
    label_path='./wavelet_feature'
    x_train = np.load(f"{path}/{args.dataset}/train.npy")
    x_test = np.load(f"{path}/{args.dataset}/test.npy")
    y_train = np.load(f"{label_path}/{args.dataset}/train_label.npy")
    y_test = np.load(f"{label_path}/{args.dataset}/test_label.npy")
    x_train = x_train.reshape(-1, 1, 59)
    x_test = x_test.reshape(-1, 1, 59)

    # Print the shapes of the datasets
    print("x_train.shape", x_train.shape)
    print("y_train.shape", y_train.shape)
    print("x_test.shape", x_test.shape)
    print("y_test.shape", y_test.shape)

    y_train, y_test = modify_labels(y_train, y_test)

    batch_size = 32
        
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    x_train, x_test = normalize_data(x_train, x_test)

    x_train = x_train.reshape(-1,59)
    x_test = x_test.reshape(-1,59)

    train_loader = DataLoader(TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long)), batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long)), batch_size=batch_size, shuffle=False, drop_last=True)

    num_epochs = args.epoch
    input_size = x_train.shape[1]
    output_size = 2
    criterion = nn.CrossEntropyLoss()

    model = DNN_Net(input_size, output_size).cuda()
    #
    meta_model = nn.Sequential(
        nn.Linear(64, 128),
        nn.BatchNorm1d(128),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(128, 64),
        nn.BatchNorm1d(64),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(64, 32),
        nn.BatchNorm1d(32),
        nn.ReLU(),

        nn.Linear(32, 2)
    )

    learning_rate = 0.001
    weight_decay = 1e-3 
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()
    minority_class, majority_class = get_minority_majority_classes(y_train)
    train_and_evaluate(model,meta_model,train_loader, test_loader, criterion, optimizer,minority_class, majority_class,clip=1,num_epochs=num_epochs)
