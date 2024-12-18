import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import random
from torch.utils.data import DataLoader, Dataset
import warnings
import matplotlib.pyplot as plt
import HM_Net_contra3 as HM_Net
import os
warnings.filterwarnings("ignore")

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

class MyDataSet(Dataset):
    def __init__(self, x_train, x_label):
        self.x_train = x_train
        self.x_label = x_label

    def __len__(self):
        return self.x_train.shape[0]

    def __getitem__(self, idx):
        return self.x_train[idx], self.x_label[idx]

def unison_shuffled_copies(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]

def normalize_data(train_x, test_x):
    for j in range(train_x.shape[2]):
        mean = np.mean(train_x[:, :, j])
        std = np.std(train_x[:, :, j])
        test_x[:, :, j] = (test_x[:, :, j] - mean) / std
        train_x[:, :, j] = (train_x[:, :, j] - mean) / std
    return train_x, test_x

def compute_loss(output, targets, reduction='mean'):
    criterion = nn.CrossEntropyLoss(reduction=reduction)
    return criterion(output, targets)

def train_and_evaluate(model, train_loader, test_loader, optimizer, num_epochs=100, conf_ratio=0.5, clip=1):
    best_test_accuracy = 0
    best_f1 = 0
    EPOCH = 0
    train_accuracy_list = []
    train_loss_list = []
    test_accuracy_list = []

    for epoch in range(num_epochs):
        model.train()
        correct = 0
        total_loss = 0.0

        for i, batch in enumerate(train_loader):
            inputs, targets = batch[0].cuda(), batch[1].cuda().view(-1).to(torch.float32)
            hidden = model.init_hidden(len(inputs))
            optimizer.zero_grad()
            output = model(inputs, hidden)
            targets = targets.long()
            loss = compute_loss(output, targets, reduction='mean')
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

        if epoch == 0:
            model.eval()
            with torch.no_grad(): 
                all_losses = []
                all_labels = []
                all_inputs = []

                for i, batch in enumerate(train_loader):
                    inputs, targets = batch[0].cuda(), batch[1].cuda().view(-1).to(torch.float32)
                    hidden = model.init_hidden(len(inputs))
                    output  = model(inputs, hidden)
                    targets = targets.long()
            

                    loss = compute_loss(output, targets, reduction='none')  
                    all_losses.extend(loss.tolist())  
                    all_labels.extend(targets.cpu().numpy().tolist())
                    all_inputs.extend(inputs.cpu().numpy())

            
            labels = [-1] * len(all_inputs)
            ratio = conf_ratio
            all_losses = np.array(all_losses)  
            all_labels = np.array(all_labels)
            all_inputs = np.array(all_inputs)

            print("all_labels", all_labels.shape)
            print("all_losses", all_losses.shape)
            print("all_inputs", all_inputs.shape)

            mask_0 = (all_labels == 0)
            loss0 = all_losses[mask_0]
            clean0 = all_inputs[mask_0]
            ind_0_sorted = np.argsort(loss0)
            len_id_0 = int(len(ind_0_sorted) * ratio)
            len_ind_0 = ind_0_sorted[:len_id_0]
            clean0 = clean0[len_ind_0] 
            label0 = all_labels[mask_0]
            IH0 = label0[len_ind_0]
            original_positions_clean0 = np.where(mask_0)[0][ind_0_sorted[:len_id_0]]
          
            for pos in original_positions_clean0:
                labels[pos] = 0
            mask_1 = (all_labels == 1)
            loss1 = all_losses[mask_1]
            clean1 = all_inputs[mask_1]
            ind_1_sorted = np.argsort(loss1)
            len_id_1 = int(len(ind_1_sorted) * ratio)
            len_ind_1 = ind_1_sorted[:len_id_1]
            clean1 = clean1[len_ind_1]  
            label1 = all_labels[mask_1]
            IH1 = label1[len_ind_1]
        
            print(np.where(mask_1)[0])
            original_positions_clean1 = np.where(mask_1)[0][ind_1_sorted[:len_id_1]]

            for pos in original_positions_clean1:
                labels[pos] = 1
            remaining0 = np.delete(all_inputs[mask_0], len_ind_0, axis=0)
            remaining1 = np.delete(all_inputs[mask_1], len_ind_1, axis=0)
            
            print("clean0 shape:", clean0.shape)
            print("clean1 shape:", clean1.shape)
            print("remaining0 shape:", remaining0.shape)
            print("remaining1 shape:", remaining1.shape)
            np.save(save_path + "/clean0.npy", clean0)
            np.save(save_path + "/clean1.npy", clean1)
            np.save(save_path + "/remaining0.npy", remaining0)
            np.save(save_path + "/remaining1.npy", remaining1)
            print("clean0 saved")
            print("clean1 saved")
            print("remaining0 saved")
            print("remaining1 saved")
                
        model.eval()
        correct = 0
        all_true_labels = []
        all_predicted_labels = []
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                test_inputs, test_targets = batch[0].cuda(), batch[1].cuda().view(-1).long()
                hidden = model.init_hidden(len(test_inputs))
                output= model(test_inputs, hidden)
                test_targets = test_targets.cpu().numpy()
                predicted = output.argmax(dim=1)
                all_true_labels.extend(test_targets)
                all_predicted_labels.extend(predicted.cpu().numpy())
                correct += np.sum(predicted.cpu().numpy() == test_targets)

        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels)
        accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        f1 = f1_score(all_true_labels, all_predicted_labels, average='weighted')
        test_accuracy_list.append(accuracy)

        if accuracy > best_test_accuracy:
            EPOCH = epoch
            best_test_accuracy = accuracy
            best_f1 = f1

        if epoch % 5 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Train Accuracy: {train_accuracy:.4f}')
            print(f'Test Accuracy: {accuracy:.4f}')
            print(f'F1 Score: {f1:.4f}')

    print(f'EPOCH: {EPOCH}')
    print(f'Best Test Accuracy: {best_test_accuracy:.4f}')
    print(f'Best F1 Score: {best_f1:.4f}')

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
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{save_path}/param.jpg")
    plt.close()

def modify_labels(y_train, y_test):
    y_train[y_train == 2] = 1
    y_test[y_test == 2] = 1
    return y_train, y_test

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and process dataset") 
    parser.add_argument("--dataset", type=str,required=True, help="Dataset to use (reRLDD or reDROZY)") 
    parser.add_argument("--epoch", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--ratio", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()
    set_seed(args.seed)
    path = "./wavelet_feature"
    save_path = f"./confident_analysis1/{args.dataset}"
    if not os.path.exists(save_path): 
        os.makedirs(save_path) 
        print(f"Save path '{save_path}' created.") 
    else: 
        print(f"Save path '{save_path}' already exists.")
    x_train = np.load(f"{path}/{args.dataset}/train_data.npy")
    x_test = np.load(f"{path}/{args.dataset}/test_data.npy")
    y_train = np.load(f"{path}/{args.dataset}/train_label.npy")
    y_test = np.load(f"{path}/{args.dataset}/test_label.npy")    
    y_train, y_test = modify_labels(y_train, y_test)
    batch_size = 32
    x_train_re = x_train.reshape(-1, 8)
    y_train_re = np.repeat(y_train, 59, axis=0)
    x_test_re = x_test.reshape(-1, 8)
    y_test_re = np.repeat(y_test, 59, axis=0)
    x_train = x_train_re.reshape(-1,1,8)
    x_test = x_test_re.reshape(-1,1,8)
    y_train = y_train_re
    y_test = y_test_re
    unique, counts = np.unique(y_train, return_counts=True)
    class_counts = dict(zip(unique, counts))
    print("Class counts in y_train:", class_counts)
    unique, counts = np.unique(y_test, return_counts=True)
    class_counts = dict(zip(unique, counts))    
    print("Class counts in y_test:", class_counts)    
    x_train, x_test = normalize_data(x_train, x_test)
    np.save(save_path +"/normalize_train.npy", x_train.reshape(-1, 8))
    np.save(save_path + "/normalize_test.npy", x_test.reshape(-1, 8))
    x_train, y_train = unison_shuffled_copies(x_train, y_train)
    train_loader = DataLoader(MyDataSet(x_train, y_train), batch_size=batch_size, shuffle=False, drop_last=False)
    test_loader = DataLoader(MyDataSet(x_test, y_test), batch_size=batch_size, shuffle=False, drop_last=False)
    num_epochs = args.epoch
    dict_size = 2
    size_list = [32, 32, 32]
    embed_size = x_train.shape[2]
    seq_len = x_train.shape[1]
    model = HM_Net.HM_Net(1.0, size_list, dict_size, embed_size, seq_len)
    model = model.cuda()
    learning_rate = 0.001
    weight_decay = 1e-4  
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_and_evaluate(model, train_loader, test_loader, optimizer, num_epochs=num_epochs,conf_ratio=args.ratio)
