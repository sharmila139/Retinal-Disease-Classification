import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import torchvision.transforms as transforms
import torch.nn as nn

classifier = nn.Linear(2048, 46)

# Define the ensemble model that combines the pre-trained models and the final layer
class EnsembleModel(nn.Module):
    def __init__(self, resnet, vgg, inception, efficientnet, densenet, classifier):
        super(EnsembleModel, self).__init__()
        self.resnet = nn.Sequential(*list(resnet.children())[:-1])
        self.vgg = nn.Sequential(*list(vgg.children())[:-1])
        self.inception = nn.Sequential(*list(inception.children())[:-1])
        self.efficientnet = efficientnet
        self.densenet = nn.Sequential(*list(densenet.children())[:-1])
        self.classifier = classifier
        
    def forward(self, x):
        x1 = self.resnet(x)
        x1 = x1.view(x1.size(0), -1)
        x2 = self.vgg(x)
        x2 = x2.view(x2.size(0), -1)
        x3 = self.inception(x)
        x3 = x3.view(x3.size(0), -1)
        x4 = self.efficientnet(x)
        x4 = x4.view(x4.size(0), -1)
        x5 = self.densenet(x)
        x5 = x5.view(x5.size(0), -1)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        x = self.classifier(x)
        return x

import pickle

# Define a dictionary to map model names to their corresponding pickle file names
model_dict = {'ResNet': 'emp-model-resnet.pkl', 'EfficientNet_B0': 'emp-model-eff.pkl', 'DenseNet': 'emp-model-dense.pkl', 'InceptionV3': 'emp-model-inception.pkl', 'VGG': 'emp-model-vgg.pkl', 'Ensemble': 'emp-model-ensemble.pkl'}
labels = ['Disease_Risk', 'DR', 'ARMD', 'MH', 'DN', 'MYA', 'BRVO', 'TSLN',
       'ERM', 'LS', 'MS', 'CSR', 'ODC', 'CRVO', 'TV', 'AH', 'ODP', 'ODE', 'ST',
       'AION', 'PT', 'RT', 'RS', 'CRS', 'EDN', 'RPEC', 'MHL', 'RP', 'CWS',
       'CB', 'ODPM', 'PRH', 'MNF', 'HR', 'CRAO', 'TD', 'CME', 'PTCR', 'CF',
       'VH', 'MCA', 'VS', 'BRAO', 'PLQ', 'HPED', 'CL']

app_mode = st.sidebar.selectbox('Select Page',['Home','Predict'])

if app_mode=='Home':
    st.title('Retinal Disease Classification')
    st.write('This project aims to classify retinal diseases using deep learning techniques.')
    st.write('The dataset used for training and testing the models contains 10,000 retinal images, with 5,000 images labeled as normal and 5,000 images labeled as diseased.')
    st.write('The project uses various Pre-trained models to predict the Retinal Disease.')


elif app_mode == 'Predict':

    st.title('Retinal Disease Classification')
    st.subheader('Select Model')
    model_name = st.selectbox('Choose a model', list(model_dict.keys()))

    st.subheader('Upload Image')
    uploaded_file = st.file_uploader('Choose an image', type=['jpg', 'jpeg', 'png'])

    if st.button('Predict'):
        if uploaded_file is not None:
            image = Image.open(uploaded_file)

            image_transforms = transforms.Compose([
                transforms.Resize((1424, 2144)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
            ])

            # Apply the transformations to the uploaded image
            input_image = image_transforms(image).unsqueeze(0)

            # Display the uploaded image
            st.image(image, use_column_width=True)

            # Perform model prediction on uploaded image
            if model_name == 'Ensemble':
                # Create an instance of the ensemble model
                picklefile = open('emp-model-resnet.pkl', 'rb')
                model = pickle.load(picklefile)
                picklefile = open('emp-model-vgg.pkl', 'rb')
                model_vgg = pickle.load(picklefile)
                picklefile = open('emp-model-inception.pkl', 'rb')
                model_inception = pickle.load(picklefile)
                picklefile = open('emp-model-eff.pkl', 'rb')
                model_eff = pickle.load(picklefile)
                picklefile = open('emp-model-dense.pkl', 'rb')
                model_dense = pickle.load(picklefile)

                model = EnsembleModel(model, model_vgg, model_inception, model_eff, model_dense, classifier)
            else:
                model_file = model_dict[model_name]
                picklefile = open(model_file, 'rb')
                model = pickle.load(picklefile)

            # Make a prediction on the image
            with torch.no_grad():
                model.eval()
                output = model(input_image)
                predicted = torch.argmax(output, dim=1).item()

            # Print the prediction
            # st.write('Prediction: {}'.format(predicted))

            if predicted is not None:
                if predicted < 46:
                    prediction_label = labels[predicted]
                    st.write('The model predicts that the retina has:', prediction_label)
                else:
                    st.write("Cannot Determine")
            else:
                st.warning('Please upload an image')