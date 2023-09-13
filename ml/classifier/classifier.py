import os
import torch
import torch.nn as nn
import torchvision
import numpy as np
import cv2

absolute_path = os.path.dirname(__file__)
device = torch.device("cpu")

class Classifier_Pneumonia:
    def __init__(self):
        self.model = torch.load(os.path.join(absolute_path, "weights/Pneumonia_transfered_on_EfficentB0.pt"), map_location=torch.device('cpu'))
        self.model = self.model.to(device)
        self.softmax = torch.nn.Softmax(dim=1)
        self.model.eval()

    def preprocessing(self, img):
        img = cv2.resize(img, (224, 224))
        if img.shape[2] == 1:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)/255.
        img =np.array(img)
        img = img.transpose(2,0,1)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = img.to(device)

        return img

    def predict(self, img):
        output = self.model(img)
        # predict = np.argmax(predict.cpu().detach().numpy(), axis=-1)
        prob = self.softmax(output)
        predict = prob[0][1]

        return predict

    def compute_prediction(self, img):

        img = self.preprocessing(img)
        pred = self.predict(img)
        
        return pred
    
class DenseNet121(nn.Module):
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121()
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, 512),
            nn.ReLU(),
            nn.Linear(512, 224),
            nn.ReLU(),
            nn.Linear(224, 14),
            nn.ReLU(),
            nn.Linear(14, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x
    
class Classifier_Aortic:
    def __init__(self):
        model = DenseNet121(1)
        model.load_state_dict(torch.load(os.path.join(absolute_path, "weights/Aortic_densenet121.pt"), map_location=torch.device('cpu')))
        self.model = model
        self.softmax = torch.nn.Softmax(dim=1)
        self.model.eval()

    def preprocessing(self, img):
        img = cv2.resize(img, (256, 256))
        if img.shape[2] == 1:
            img = np.dstack([img, img, img])
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)/255.
        img =np.array(img)
        img = img.transpose(2,0,1)
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img)
        img = img.to(device)

        return img

    def predict(self, img):
        predict = self.model(img)
        return predict

    def compute_prediction(self, img):

        img = self.preprocessing(img)
        pred = self.predict(img)
        pred = pred.detach().numpy()
        
        return pred[0][0]