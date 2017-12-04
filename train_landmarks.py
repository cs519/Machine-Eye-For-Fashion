import time
from tqdm import tqdm
import torch
from torch.autograd import Variable
from torch import optim
from preprocessing.dataloader import FashionLandmarkDataset
from model.FashionNet import landmark_network
from torchvision import transforms
from loss.CustomLoss import CustomLoss


def train_model(model,
                criterion,
                optimizer,
                train_dataloader,
                valid_dataloader=None,
                num_epochs=25,
                use_gpu=None):

    """
    Train the landmarks locations network
    :param model: pytorch model to use
    :param criterion: loss function for the model
    :param optimizer: optimizer for model
    :param train_dataloader: Dataset loader for train data
    :param valid_dataloader: Dataset loader for validation data; default: None
    :param num_epochs: number of epochs to train the model; default: 25
    :param use_gpu: boolean on whether or not use GPU; default: None
    """
    # Get the start time of training
    since = time.time()

    # Save the current state of the model, current best is 0
    best_model_weights = model.state_dict()
    best_loss = None

    # Check if there is GPU available, if so, use it
    if use_gpu is None:
        use_gpu = torch.cuda.is_available()

    # tqdm is a progress bar
    # train model for num_epochs
    for epoch in tqdm(range(num_epochs)):
        # print current epoch to console
        tqdm.write('Epoch {}/{}'.format(epoch, num_epochs - 1))
        tqdm.write('-' * 10)

        # for each epoch, there is a train phase followed by validation phase
        for phase in ['train', 'valid']:
            # Set model in train or validation mode
            # Load data from corresponding dataloader
            if phase == 'train':
                model.train(True)
                dset_loader = train_dataloader
            else:
                model.train(False)
                dset_loader = valid_dataloader

            # Get dataset size
            dataset_sizes = len(dset_loader)

            # initialize the loss
            running_loss = 0.0

            # Iterate over data
            # for data in dset_loader:
            for i in range(dataset_sizes):
                # get inputs and labels
                # inputs, landmarks, visibility = data
                inputs, landmarks, visibility = dset_loader[i]

                # Wrap data in Variables
                if use_gpu:
                    # load onto GPU if available
                    inputs = Variable(inputs.cuda())
                    landmarks = Variable(landmarks.cuda())
                    visibility = Variable(visibility.cuda())
                else:
                    inputs = Variable(inputs)
                    landmarks = Variable(landmarks)
                    visibility = Variable(visibility)

                # zero the parameter gradients
                optimizer.zero_grad()

                # foward pass
                outputs = model(inputs)
                l_preds, vis_preds = outputs
                # calculate loss
                loss = criterion(outputs, (landmarks, visibility))

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                # statistics
                # sum the loss for the epoch
                running_loss += loss.data[0]

            # loss for the epoch
            epoch_loss = running_loss / dataset_sizes


            # print loss and accuracy
            tqdm.write('{} Loss: {:.4f}'.format(
                phase, epoch_loss))
            # if the current model has the lowest loss then, save this model
            # if this the first epoch, save this model
            if phase == 'valid' and (epoch_loss < best_loss or best_loss is None):
                best_loss = epoch_loss
                best_model_weights = model.state_dict()

        print()
    
    # print the time it took to train
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    # print best validation loss for model
    print('Best val loss: {:4f}'.format(best_loss))
    # save the model with ite best weights
    model.load_state_dict(best_model_weights)

    return model


if __name__ == '__main__':
    import os

    # set the data directories
    train_data_dir = os.path.join(os.getcwd(),
                                  'data/DeepFashion/train.csv')
    valid_data_dir = os.path.join(os.getcwd(), 'data/DeepFashion/valid.csv')
    train_dir = os.path.join(os.getcwd(), 'data/DeepFashion/train')
    valid_dir = os.path.join(os.getcwd(), 'data/DeepFashion/valid')

    # load training and validation images and targets
    train_loader = FashionLandmarkDataset(
        train_data_dir,
        train_dir,
        # transform data (similar to Keras' ImageDataGenerator)
        # resize so the width is 226
        # crop from center, to make image have height of 400, width 266
        # randomly horizontalflip for data augmentation 
        # put the image in a tensor
        # normalize the image
        transform=transforms.Compose([
            transforms.Resize(266),
            transforms.CenterCrop(400,266),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

    valid_loader = FashionLandmarkDataset(
        valid_data_dir,
        valid_dir,
        # transform data (similar to Keras' ImageDataGenerator)
        # resize so the width is 226
        # crop from center, to make image have height of 400, width 266
        # put the image in a tensor
        # normalize the image
        transform=transforms.Compose([
            transforms.Resize(266),
            transforms.CenterCrop(400,266),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]))

    # create the landmark network
    model_ft = landmark_network(num_outputs=6)
    # use the custom loss which sums cross entropy and euclidean loss
    criterion_ft = CustomLoss()
    # use Adam to optimize
    # Since the pretrained layers won't have their weights udpated, use the lambda to choose only parameters that will be optimized
    optimizer_ft = optim.Adam(
        filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1e-2)

    # train the model on the GPU for 1000 epochs
    model_ft = train_model(
        model_ft,
        criterion_ft,
        optimizer_ft,
        train_loader,
        valid_dataloader=valid_loader,
        num_epochs=1000,
        use_gpu=True)
