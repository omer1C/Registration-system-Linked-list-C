import pandas
import numpy as np
import matplotlib.pyplot as plt

import functions
import torch
import torch.nn as nn
import torch.optim as optim
import sys
from pathlib import Path
import pickle

from PIL import Image
path = Path('/Users/omercohen/PycharmProjects/Assigenment3/')

#---section 2---#
'''
In case we want to use google colab
'''
# from google.colab import drive
# drive_path = '/content/gdrive/MyDrive/' # TODO - UPDATE ME!
# drive.mount('/content/gdrive')
#---section 3---#
import glob
'''
In this section we arrange the data in a way are cnn 
can use efficiently
'''
def data_prepare(data_path):
  images = []
  new_path = '/Users/omercohen/PycharmProjects/Assigenment3/'+data_path
  for file in Path(new_path).iterdir():
      #print(file)
      filename = str(file).split("/")[-1]            # get the name of the .jpg file
      img = np.asarray(Image.open(file))        # read the image as a numpy array
      images.append([filename,((np.array(img)[:, :, :3]/ 255.0) - 0.5)]) # add and normelize the data
  sorted_data = sorted(images, key=lambda x: x[0])
  data = []
  person = []
  pair_of_shoes = []
  iter = 0
  side = 0

  for pictur in sorted_data:
      pair_of_shoes.append(np.array(pictur[1]))                                        # read the image as a numpy array
      iter += 1
      side += 1
      if side == 2 :    # add pair of shoes  to the person
        side = 0
        person.append(np.array(pair_of_shoes))
        pair_of_shoes = []
      if iter%6 == 0 : # add the person to the list of data
        data.append(np.array(person))
        person = []
  return data


data_path_set = ["data/train","data/test_m/","data/test_w/"]

train_val_data = data_prepare(data_path_set[0])
train_precent = 0.80
split_index = int(train_precent*len(train_val_data)) #index to spit training & validation into 80 % and 20 %

train_data = np.array(train_val_data[:split_index])
valid_data = np.array(train_val_data[split_index:])

test_m_data  = np.array(data_prepare(data_path_set[1]))
test_w_data  = np.array(data_prepare(data_path_set[2]))

#---section 4---#
'''
sanity check we prepare the data correctly 
'''
# Run this code
# test1 = np.load(path+"data/preproccessing_validation/preproccessing_validation_test1.npy")
# test2 = np.load(path+"data/preproccessing_validation/preproccessing_validation_test2.npy")
# test3 = np.load(path+"data/preproccessing_validation/preproccessing_validation_test3.npy")
# if (test1 != train_data[4,0,0,:,:,:]).any() or \
#    (test2 != train_data[4,0,1,:,:,:]).any() or \
#    (test3 != train_data[4,1,1,:,:,:]).any():
#    print("preprocessing error, make sure you followed all instructions carefully")

plt.figure()
plt.imshow(train_data[4,0,0,:,:,:]+0.5) # left shoe of first pair submitted by 5th student
#plt.show()
plt.figure()
plt.imshow(train_data[4,0,1,:,:,:]+0.5) # right shoe of first pair submitted by 5th student
plt.figure()
plt.imshow(train_data[4,1,1,:,:,:]+0.5) # right shoe of second pair submitted by 5th student

#---section 4---#
# generate same pair of shoes
def generate_same_pair(data):
  same_pair_data = []
  for student in data:
    for pic in student:
      picl = pic[0]
      picr = pic[1]
      same_pair_data.append(np.concatenate((picl, picr), axis=0))
  return np.array(same_pair_data)

# Run this code
print(train_data.shape) # if this is [N, 3, 2, 224, 224, 3]
print(generate_same_pair(train_data).shape) # should be [N*3, 448, 224, 3]
plt.imshow(generate_same_pair(train_data)[0]+0.5) # should show 2 shoes from the same pair

#---section 5---#
# generate different pair of shoes
def generate_different_pair(data):
  different_pair_data = []
  for student in range(len(data)) :
    different_pair_data.append(np.concatenate((data[student][0][0], data[student][1][1]), axis=0))
    different_pair_data.append(np.concatenate((data[student][1][0], data[student][2][1]), axis=0))
    different_pair_data.append(np.concatenate((data[student][2][0], data[student][0][1]), axis=0))
  return np.array(different_pair_data)
# Run this code
print(train_data.shape) # if this is [N, 3, 2, 224, 224, 3]
print(generate_different_pair(train_data).shape) # should be [N*3, 448, 224, 3]
plt.imshow(generate_different_pair(train_data)[0]+0.5) # should show 2 shoes from different pairs

#---section 6---#
#in case of using google colab
# from google.colab import drive
# drive.mount('/content/drive')

#---section 7---#
def get_accuracy(model, data, batch_size=50,device='cpu'):
    """Compute the model accuracy on the data set. This function returns two
    separate values: the model accuracy on the positive samples,
    and the model accuracy on the negative samples.
    """

    model.eval()
    n = data.shape[0]

    data_pos = generate_same_pair(data)      # should have shape [n * 3, 448, 224, 3]
    data_neg = generate_different_pair(data) # should have shape [n * 3, 448, 224, 3]

    pos_correct = 0
    for i in range(0, len(data_pos), batch_size):
        xs = torch.Tensor(data_pos[i:i+batch_size]).permute(0,3,1,2)
        xs = xs.to(device)
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        pred = pred.detach().cpu().numpy()
        pos_correct += (pred == 1).sum()

    neg_correct = 0
    for i in range(0, len(data_neg), batch_size):
        xs = torch.Tensor(data_neg[i:i+batch_size]).permute(0,3,1,2)
        xs = xs.to(device)
        zs = model(xs)
        pred = zs.max(1, keepdim=True)[1] # get the index of the max logit
        pred = pred.detach().cpu().numpy()
        neg_correct += (pred == 0).sum()

    return pos_correct / (n * 3), neg_correct / (n * 3)

#---section 8---#
'''
This section is for training the model
'''
def train_model(model,
                train_data=train_data,
                validation_data=valid_data,
                batch_size=32,
                learning_rate=0.001,
                weight_decay=1e-4,
                epochs=30,
                checkpoint_path="/Users/omercohen/PycharmProjects/ImageClassification/",
                model_name=""):
    # Initialize Loss function and optimizer

    # Loss function
    criterion = nn.CrossEntropyLoss()
    # Optimaizer
    optimaizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Create the positive and negative pairs
    positive_pair_train = generate_same_pair(train_data)
    negative_pair_train = generate_different_pair(train_data)

    positive_pair_val = generate_same_pair(validation_data)
    negative_pair_val = generate_different_pair(validation_data)
    val_labels = torch.cat((torch.ones(len(positive_pair_val)), torch.zeros(len(negative_pair_val)))).long()

    accuracy_train_list = []
    loss_train_list = []
    accuracy_val_list = []
    loss_val_list = []
    best_epoch_accuracy = 0

    for epoch in range(epochs):
        # shuffle the positive and negative pairs
        num_of_pairs = positive_pair_train.shape[0]
        shuffeled_index = torch.randperm(num_of_pairs)
        positive_shuffeld_pair = positive_pair_train[shuffeled_index]
        negative_shuffeld_pair = negative_pair_train[shuffeled_index]

        runing_epoch_loss = 0
        accuracy_epoch = 0
        total_train = 0


        for i in range(0, len(positive_pair_train)-(batch_size // 2), batch_size // 2):
            # sample batch_size//2 of positive pairs and batch_size//2 of negative pairs
            positive_batch_train = positive_shuffeld_pair[i:i + (batch_size // 2)]
            negative_batch_train = negative_shuffeld_pair[i:i + (batch_size // 2)]
            all_batch_pairs = np.concatenate([positive_batch_train,negative_batch_train])
            # Create the labels of the batch and combine the positive and negative half batches
            num_of_ones = batch_size // 2
            num_of_zeros = batch_size // 2
            labels = torch.cat((torch.ones(num_of_ones), torch.zeros(num_of_zeros))).long()
            all_batch_pairs = torch.from_numpy(all_batch_pairs).permute(0, 3, 1, 2).float()
            # Conversion from numpy array to torch tensor (if working with gpu also change device to gpu)
            # ...

            # ... # Reset the optimizer
            optimaizer.zero_grad()
            # ... # Predict output
            predictions = model(all_batch_pairs)
            # ... # Compute the loss
            loss = criterion(predictions, labels)
            # ... # Backward pass
            loss.backward()
            # ... # Update the parameters
            optimaizer.step()

            runing_epoch_loss += loss.item()
            accuracy_epoch += ((predictions.argmax(dim=1) == labels).sum().item())
            total_train += labels.size(0)

        # track the accuracy and loss of the training and validation
        # ...
        # calculate loss and accuracy each epoch for the train
        epoch_train_loss = runing_epoch_loss / (len(positive_pair_train)//(batch_size // 2))
        loss_train_list.append(epoch_train_loss)
        epoch_train_accuracy = accuracy_epoch / total_train
        accuracy_train_list.append(epoch_train_accuracy)

        # calculate loss and accuracy each epoch for the validation

        val_all_pairs = np.concatenate([positive_pair_val,negative_pair_val])
        val_all_pairs = torch.from_numpy(val_all_pairs).permute(0, 3, 1, 2).float()
        val_predictions = model(val_all_pairs)

        val_epoch_loss = criterion(val_predictions, val_labels)
        val_epoch_accuracy = (val_predictions.argmax(dim=1) == val_labels).sum().item() / val_labels.size(0)

        loss_val_list.append(val_epoch_loss.item())
        accuracy_val_list.append(val_epoch_accuracy)

        # print epoch results :
        print(
            f"Epoch [{epoch + 1}/{epochs}], Validation Loss: {val_epoch_loss:.4f}, Validation Accuracy: {100 * val_epoch_accuracy:.2f}%")
        print(
            f"Epoch [{epoch + 1}/{epochs}], Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {100 * epoch_train_accuracy:.2f}%")
        # checkpoint the model
        # ...
        if val_epoch_accuracy * 100 > best_epoch_accuracy:
            best_epoch_accuracy = float("{:.2f}".format(val_epoch_accuracy))*100
            best_epoch = epoch
            best_model_param = model.state_dict()
            if model_name == 'CNN':
                torch.save(best_model_param,
                           checkpoint_path + "best_CNN_model.pk")
                if best_epoch_accuracy ==84:
                    learning_rate = learning_rate/2
            elif model_name == 'CNNChannel':
                torch.save(best_model_param,
                           checkpoint_path + "best_CNNChannel_model.pk")
                if best_epoch_accuracy == 87:
                    learning_rate = learning_rate/2
            print(f"New best epoch, epoch number {best_epoch+1:.4f} with accuracy: {best_epoch_accuracy:.4f} ")

    return  loss_train_list, loss_val_list , accuracy_train_list, accuracy_val_list, best_model_param



#---section 9---#
'''
sanity check that the model can learn 
the data we put in training using the same data in the validation,
we will want to see the same behavior of loss and accuracy in the validation
'''
train_data_sanity_check = train_data[0:6]
valid_data_sanity_check = train_data_sanity_check

modelCNNChannel = functions.CNNChannel()
train_losses_CNNChannel, valid_losses_CNNChannel, train_accs_CNNChannel, valid_accs_CNNChannel = train_model(modelCNNChannel, train_data_sanity_check,
                                                                      valid_data_sanity_check, batch_size = 2,
                                                                      learning_rate=0.001,
                                                                      weight_decay=0.0001, epochs=28)


modelCNN = functions.CNN()
train_losses_CNN, valid_losses_CNN, train_accs_CNN, valid_accs_CNN = train_model(modelCNN, train_data_sanity_check,
                                                                     valid_data_sanity_check, batch_size = 2, learning_rate = 0.001,
                                                                     weight_decay=0.0001, epochs=28)

epochs = list(range(28))
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(epochs, valid_losses_CNNChannel)
axs[0, 0].plot(epochs, train_losses_CNNChannel)
axs[0, 0].set_title('CNN Channel Loss Val & Train')
axs[0,0].set(xlabel='Epochs', ylabel='Loss')
axs[0, 1].plot(epochs, valid_accs_CNNChannel)
axs[0, 1].plot(epochs, train_accs_CNNChannel)
axs[0, 1].set_title('CNN Channel Accuracy Val & Train')
axs[0,1].set(xlabel='Epochs', ylabel='Accuracy')
axs[1, 0].plot(epochs, valid_losses_CNN)
axs[1, 0].plot(epochs, train_losses_CNN)
axs[1, 0].set_title('CNN Loss Val & Train')
axs[1,0].set(xlabel='Epochs', ylabel='Loss')
axs[1, 1].plot(epochs, valid_accs_CNN)
axs[1, 1].plot(epochs, train_accs_CNN)
axs[1, 1].set_title('CNN Accuracy Val & Train')
axs[1,1].set(xlabel='Epochs', ylabel='Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

###---section 10---###
# Training
CNN_model = functions.CNN()
CNNChannel_model = functions.CNNChannel()

# Insert your training here
#checkpoint_path  = drive_path+"checkpoints/"

# Paraameters :

modelCNNChannel = functions.CNNChannel()
train_losses_CNNChannel, valid_losses_CNNChannel, train_accs_CNNChannel, valid_accs_CNNChannel, best_model_param_CNNchannel = train_model(modelCNNChannel, train_data,
                                                                      valid_data, batch_size=42,
                                                                      learning_rate=0.001,
                                                                      weight_decay=0.0001, epochs=30, model_name='CNNChannel')


#testing the model:
modelCNN = functions.CNN()
train_losses_CNN, valid_losses_CNN, train_accs_CNN, valid_accs_CNN, best_model_param_CNN = train_model(modelCNN, train_data,
                                                                     valid_data, batch_size=42, learning_rate=0.001,
                                                                     weight_decay=0.0001, epochs=30, model_name='CNN')

best_model_param_CNN = torch.load('best_CNN_model.pk')
best_model_param_CNNchannel = torch.load('best_CNNChannel_model.pk')
#---Testing---#
best_CNN_model = functions.CNN()
best_CNN_model.load_state_dict(best_model_param_CNN)

best_CNNChannel_model = functions.CNNChannel()
best_CNNChannel_model.load_state_dict(best_model_param_CNNchannel)

pos_men_CNN = 0 ; neg_men_CNN = 0
pos_women_CNN = 0 ; neg_women_CNN = 0
pos_men_CNNChannel = 0 ; neg_men_CNNChannel = 0
pos_women_CNNChannel = 0 ; neg_women_CNNChannel = 0

pos_men_CNN, neg_men_CNN = get_accuracy(best_CNN_model, test_m_data)
pos_women_CNN, neg_women_CNN = get_accuracy(best_CNN_model, test_w_data)
pos_men_CNNChannel, neg_men_CNNChannel = get_accuracy(best_CNNChannel_model, test_m_data)
pos_women_CNNChannel, neg_women_CNNChannel = get_accuracy(best_CNNChannel_model, test_w_data)

print("CNN: Positive accuracy over the men's test dataset: {:.2f}%, Negative accuracy: {:.2f}%, Mean {:.2f}%".format(100 * pos_men_CNN, 100 * neg_men_CNN,100 * pos_men_CNN/2+ 100 * neg_men_CNN/2))
print("CNN: Positive accuracy over the women's test dataset: {:.2f}%, Negative accuracy: {:.2f}%, Mean {:.2f}%".format(100 * pos_women_CNN, 100 * neg_women_CNN,100 * pos_women_CNN/2+ 100 * neg_women_CNN/2))
print("Mean accuracf for CNN is: {:.2f}%".format(100 * pos_men_CNN/4+ 100 * neg_men_CNN/4+100*pos_women_CNN/4+ 100 * neg_women_CNN/4))
print("CNNChannel: Positive accuracy over the men's test dataset: {:.2f}%, Negative accuracy: {:.2f}%, Mean {:.2f}%".format(100 * pos_men_CNNChannel, 100 * neg_men_CNNChannel,100 * pos_men_CNNChannel/2+ 100 * neg_men_CNNChannel/2))
print("CNNChannel: Positive accuracy over the women's test dataset: {:.2f}%, Negative accuracy: {:.2f}%, Mean {:.2f}%".format(100 * pos_women_CNNChannel, 100 * neg_women_CNNChannel,100 * pos_women_CNNChannel/2+ 100 * neg_women_CNNChannel/2))
print("Mean accuracf for CNNChannel is: {:.2f}%".format(100 * pos_men_CNNChannel/4+ 100 * neg_men_CNNChannel/4+100*pos_women_CNNChannel/4+ 100 * neg_women_CNNChannel/4))


#--------------#

epochs = list(range(30))
fig, axs = plt.subplots(2, 2)
axs[0, 0].plot(epochs, valid_losses_CNNChannel)
axs[0, 0].plot(epochs, train_losses_CNNChannel)
axs[0, 0].set_title('CNN Channel Loss Val & Train')
axs[0,0].set(xlabel='Epochs', ylabel='Loss')
plt.grid(True)
axs[0, 1].plot(epochs, valid_accs_CNNChannel)
axs[0, 1].plot(epochs, train_accs_CNNChannel)
axs[0, 1].set_title('CNN Channel Accuracy Val & Train')
axs[0,1].set(xlabel='Epochs', ylabel='Accuracy')
plt.grid(True)
axs[1, 0].plot(epochs, valid_losses_CNN)
axs[1, 0].plot(epochs, train_losses_CNN)
axs[1, 0].set_title('CNN Loss Val & Train')
axs[1,0].set(xlabel='Epochs', ylabel='Loss')
plt.grid(True)
axs[1, 1].plot(epochs, valid_accs_CNN)
axs[1, 1].plot(epochs, train_accs_CNN)
axs[1, 1].set_title('CNN Accuracy Val & Train')
axs[1,1].set(xlabel='Epochs', ylabel='Accuracy')
plt.grid(True)
plt.tight_layout()
plt.show()

#---------#
'''
Example for image that classified correctly
and example for image that classified incorrectly
'''
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # Ignore the warning about clipping input data

pos_test_m_data = generate_same_pair(test_m_data)
# Make inputs
pos_test_m_data_reshaped = pos_test_m_data.reshape(-1, 448, 224, 3)
input_tensor = torch.from_numpy(pos_test_m_data_reshaped).permute(0, 3, 1, 2).float()

label = torch.ones(1).long()

fig, axes = plt.subplots(1, 1, figsize=(20, 20))

i = 0
flag = False

print('Positive pair correctly classified')
while not flag and i < len(input_tensor):
    pred = best_CNNChannel_model(input_tensor[i].unsqueeze((0)))
    accuracy = ((pred.argmax(dim=1) == label).sum().item()) / label.size(0)
    if accuracy == 1 :
        print(i)
        axes[0].imshow(np.transpose(input_tensor[i].numpy(), (1, 2, 0)))
        axes[0].set_title("Positive pair correctly classified", fontsize=15)
        flag = True
    i += 1

print('Positive pair incorrectly classified')
i = 0
flag = False
fig, axes = plt.subplots(1, 1, figsize=(20, 20))

while not flag and i < len(input_tensor):
    pred = best_CNNChannel_model(input_tensor[i].unsqueeze((0)))
    accuracy = ((pred.argmax(dim=1) == label).sum().item()) / label.size(0)
    if accuracy == 0 :
        print(i)
        axes[0].imshow(np.transpose(input_tensor[i].numpy(), (1, 2, 0)))
        axes[0].set_title("Positive pair incorrectly classified", fontsize=15)
        flag = True
    i += 1
    

