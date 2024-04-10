import copy
import math
import random

import os

from PIL import Image
import tracery
from techniques import *
import cv2
from scipy.spatial import distance as dist

from settings import *

from meanDiffModel.useMeanToClassifyTestTensorSingleImageFinal import Net

import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

def score_art_tf_local(_path):
    #Load the saved means
    tensor_folder = './meanDiffModelV2/'
    load_path = os.path.join(tensor_folder, 'mean_features_by_label.pt')

    # Load the dictionary containing the mean features
    mean_features_by_label = torch.load(load_path)
    #print(f"Mean features by label have been loaded from {load_path}")

    net = Net()

    # modelLocation = './meanDiffModel/paintingVsSculpture.pth'
    modelLocation = './meanDiffModelV2/artVsRandomNoise.pth'
   
    net.load_state_dict(torch.load(modelLocation, map_location=torch.device('cpu')))

    transform = transforms.Compose(
        [transforms.Resize((64,64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    for p, folders, files in os.walk(_path):
        for _file in files:
            if _file.endswith(".png"): 
                image = Image.open(os.path.join(p,_file))

                # handle alpha issue 
                if image.mode == 'RGBA':
                    # Drop the alpha channel
                    image = image.convert('RGB')

                image = transform(image)    # Apply the transformation
                image = image.unsqueeze(0)  # Add batch dimension

                # Extract means directly by keys if you know them
                artMean = mean_features_by_label['art']
                notArtMean = mean_features_by_label['notArt']

                # since we're not training, we don't need to calculate the gradients for our outputs
                with torch.no_grad():
                    outputs, tensor_before_fc = net(image)
                    artDifference = torch.sum(torch.abs(tensor_before_fc - artMean))
                    notArtDifference = torch.sum(torch.abs(tensor_before_fc - notArtMean))

                    #print("ART MEANS", p._id, artDifference, notArtDifference, artMean, notArtMean)

                    if(artDifference < notArtDifference):
                        #predict art
                        myPredicted = 0
                    else:
                        #predict not art
                        myPredicted = 1

                    # For classification using the CNN, the class with the highest energy is the class chosen for prediction
                    _, predicted = torch.max(outputs.data, 1)

                    #If this outputs 0 if the NN thinks its art, 1 if not art 
                    #print('0 is art, 1 is not art')
                    if myPredicted == 0:
                        print(f'{p,_file} -> art')
                    else:
                        print(f'{p,_file} -> not art')

# score_art_tf_local('./art-only-comparison/')
# score_art_tf_local('./paper-img/')
score_art_tf_local('./ase-favorites/')