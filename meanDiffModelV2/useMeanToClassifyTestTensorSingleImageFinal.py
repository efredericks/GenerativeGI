import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
#import torch.optim as optim
from PIL import Image
#import matplotlib.pyplot as plt
#import numpy as np
import multiprocessing
import os


class Net(nn.Module):

        def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 6, 5)
                self.pool = nn.MaxPool2d(2, 2)
                self.conv2 = nn.Conv2d(6, 16, 5)
                self.fc1 = nn.Linear(16 * 13 * 13, 120)
                self.fc2 = nn.Linear(120, 84)
                self.fc3 = nn.Linear(84, 2)

        def forward(self, x):
                x = self.pool(F.relu(self.conv1(x)))
                x = self.pool(F.relu(self.conv2(x)))
                
                x = torch.flatten(x, 1) # flatten all dimensions except batch
                tensor_before_fc = x.clone() # Flatten it and then return it

                x = F.relu(self.fc1(x))
                x = F.relu(self.fc2(x))
                x = self.fc3(x)

                return x, tensor_before_fc

def main():

    #Load the saved means
    tensor_folder = '/home/bobeldyd/programming/python/artCollab/v2'
    load_path = os.path.join(tensor_folder, 'mean_features_by_label.pt')

    # Load the dictionary containing the mean features
    mean_features_by_label = torch.load(load_path)
    print(f"Mean features by label have been loaded from {load_path}")

    net = Net()

    modelLocation = '/home/bobeldyd/programming/python/artCollab/v2/artVsRandomNoise.pth'
   
    net.load_state_dict(torch.load(modelLocation, map_location=torch.device('cpu')))

    transform = transforms.Compose(
        [transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    # Load specific image
    #image_path = "./artImage.jpg"
    #image_path = "./notArtImage.jpg"
    #image_path = "./noise_10.png"
    image_path = "./0006.jpg"
    
    image = Image.open(image_path)
    image = transform(image)    # Apply the transformation
    image = image.unsqueeze(0)  # Add batch dimension

    # Extract means directly by keys if you know them
    artMean = mean_features_by_label['art']
    notArtMean = mean_features_by_label['notArt']

    with torch.no_grad():
        outputs, tensor_before_fc = net(image)
        artDifference = torch.sum(torch.abs(tensor_before_fc - artMean))
        notArtDifference = torch.sum(torch.abs(tensor_before_fc - notArtMean))

        if(artDifference < notArtDifference):
            #predict art
            myPredicted = 0
        else:
            #predict not art
            myPredicted = 1

        # For classification using the CNN, the class with the highest energy is the class chosen for prediction
        _, predicted = torch.max(outputs.data, 1)

    #0: NN thinks its art; 1: not art 
    print('0 is art, 1 is not art')
    print(f'Predicted = {predicted}')


    if(artDifference < notArtDifference):
         print('Closer to the art mean')
    else:
         print('Closer to the not art mean')

    tensor_file_path = "/home/bobeldyd/programming/python/artCollab/v2/tensorGenerated.pt"  # replace with your desired file path if this is even needed
    torch.save(tensor_before_fc, tensor_file_path)
    print(f"Tensor saved to {tensor_file_path}")


if __name__ == '__main__':
    multiprocessing.freeze_support()
    print ("Running script to calculate the tensor generated from the classifier model")
    main()
