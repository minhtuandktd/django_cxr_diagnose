from rest_framework.views import APIView
from django.core.files.storage import FileSystemStorage
from rest_framework.response import Response
from django.conf import settings
from django.utils.datastructures import MultiValueDictKeyError

import cv2
import torch
import numpy as np
from torchvision import models
import torch.nn as nn
from ml.classifier.classifier import Classifier_Pneumonia, Classifier_Aortic
from ml.valid_input.features_extract import transform_features, ExtractFeatures
import json
import requests
from PIL import ImageDraw, Image 
from io import BytesIO
import dicom2jpg
import random

# Load features extract to calculate distance
# features_extract = ExtractFeatures()
# features_extract.load_state_dict(torch.load('./ml/valid_input/vgg16_features_extract.pth'))
# print('Load features extract success!')
# img_org_1 = cv2.imread("./ml/valid_input/images_compare/IM-0584-0001.jpeg")
# img_org_2 = cv2.imread("./ml/valid_input/images_compare/NORMAL2-IM-0516-0001.jpeg")
# img_org_3 = cv2.imread("./ml/valid_input/images_compare/person1_bacteria_1.jpeg")
# img_org_4 = cv2.imread("./ml/valid_input/images_compare/person1520_virus_2647.jpeg")
# img_org_1 = transform_features(img_org_1)
# img_org_2 = transform_features(img_org_2)
# img_org_3 = transform_features(img_org_3)
# img_org_4 = transform_features(img_org_4)
# vector1 = features_extract(img_org_1)
# vector2 = features_extract(img_org_2)
# vector3 = features_extract(img_org_3)
# vector4 = features_extract(img_org_4)
# vector1 = vector1.detach().numpy()
# vector2 = vector2.detach().numpy()
# vector3 = vector3.detach().numpy()
# vector4 = vector4.detach().numpy()
# vector1 = vector1 / np.linalg.norm(vector1)
# vector2 = vector2 / np.linalg.norm(vector2)
# vector3 = vector3 / np.linalg.norm(vector3)
# vector4 = vector4 / np.linalg.norm(vector4)
# print('Calculate base vector done!')

# Load model classifier input valid
classifier_input = models.vgg16()
classifier_input.classifier = nn.Sequential(
    nn.Linear(25088, 4096),
    nn.ReLU(),
    nn.Linear(4096, 1000),
    nn.ReLU(),
    nn.Linear(1000, 512),
    nn.ReLU(),
    nn.Linear(512, 224),
    nn.ReLU(),
    nn.Linear(224, 2)
)
classifier_input.load_state_dict(torch.load('./ml/valid_input/vgg16_classifier.pth', map_location=torch.device('cpu')))
print('Load classifier input valid success!')

# Load prediction models
classifier_pneumonia = Classifier_Pneumonia()
classifier_aortic = Classifier_Aortic()


class CustomFileSystemStorage(FileSystemStorage):
    def get_available_name(self, name, max_length=None):
        self.delete(name)
        return name
    
class home(APIView):
    def get(self, request):
        return Response("Hello World!")
    
class call_model(APIView):

    def get(self, request):
        return Response("Hello World ChestX-ray!")

    def post(sefl, request):
        fss = CustomFileSystemStorage()
        try:
            image = request.FILES["image"]
            # print("Name", image.file)
            _image = fss.save(f"{settings.UPLOADS_ROOT}/{image.name}", image)
            path = str(settings.UPLOADS_ROOT) + "/" + image.name
            # Read the image
            img=cv2.imread(path)

            # Valid input
            # Use distance to valid stage1:
            # img_vld = transform_features(img)
            # vector = features_extract(img_vld)
            # vector = vector.detach().numpy()
            # vector = vector / np.linalg.norm(vector)
            # distance = (np.linalg.norm(vector1 - vector, axis=1) + np.linalg.norm(vector2 - vector, axis=1) 
            #                 + np.linalg.norm(vector3 - vector, axis=1) + np.linalg.norm(vector4 - vector, axis=1)) / 4   
            # if distance > 0.89:
            #     return Response({
            #         'code':200,
            #         'input_valid':'no'
            #     })
            
            # Use classifier to valid stage2:
            img_cls = transform_features(img)
            img_cls = torch.unsqueeze(img_cls, 0)
            output_cls = classifier_input(img_cls)
            _, pred_cls = torch.max(output_cls, 1)
            if pred_cls[0] == 0:
                return Response({
                    'code':200,
                    'input_valid':"no"
                })
            
            # Prediction
            pred_pneumonia = classifier_pneumonia.compute_prediction(img)
            pred_aortic = classifier_aortic.compute_prediction(img)
            # if prediction == 0:
            #     return Response({
            #         "code": 200,
            #         "input_valid": "yes",
            #         "prediction": "Normal"
            #         })
            # if prediction == 1:
            #     return Response({
            #         "code": 200,
            #         "input_valid": "yes",
            #         "prediction": "Pneumonia"
            #         })

            return Response({
                    "code": 200,
                    "input_valid": "yes",
                    "pred_pneumonia": pred_pneumonia,
                    "pred_aortic": pred_aortic
                    })
            
        except MultiValueDictKeyError:
            return Response({
                "code":400,
                "message":"No image selected!"
            })
        except TypeError:
            return Response({
                "code":415,
                "message": "File type not allowed! You must upload images type: jpg, png, jpeg,..."
            })
        except:
            return Response({
                "code":500,
                "message":"Smt bad has been occured!"
            })


class call_model_link(APIView):

    def get(self, request):
        return Response("Hello World ChestX-ray!")

    def post(sefl, request):
        json_post = json.loads(request.body)
        link_image = json_post['link_image']
        if link_image.find("dcm") != -1 :
            img_raw = requests.get(link_image)
            raw_bytes = img_raw.content
            img = dicom2jpg.io2img(BytesIO(raw_bytes))
            if len(img.shape) == 2:
                cvImage = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            elif img.shape[2] == 4:
                cvImage = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            else:
                cvImage = img
        elif (link_image.find("jpg") != -1) or (link_image.find("png") != -1) or (link_image.find("jpeg") != -1):
            img_url = requests.get(link_image)
            raw_data = img_url.content
            img = Image.open(BytesIO(raw_data))
            img_arr = np.array(img)
            if len(img_arr.shape) == 2:
                cvImage = cv2.cvtColor(img_arr, cv2.COLOR_GRAY2RGB)
            elif img_arr.shape[2] == 4:
                cvImage = cv2.cvtColor(img_arr, cv2.COLOR_RGBA2RGB)
            else:
                cvImage = img_arr

        try:

            name = str(random.randint(1, 100)) + ".png"
            path = str(settings.UPLOADS_ROOT) + "/" + name
            print(path)
            cv2.imwrite(path, cvImage)
            path_return = "/uploads/" + name
            
            # Use classifier to valid stage2:
            img_cls = transform_features(cvImage)
            img_cls = torch.unsqueeze(img_cls, 0)

            output_cls = classifier_input(img_cls)
            _, pred_cls = torch.max(output_cls, 1)
            if pred_cls[0] == 0:
                return Response({
                    'code':200,
                    'input_valid':"no"
                })
            
            # Prediction
            pred_pneumonia = classifier_pneumonia.compute_prediction(cvImage)
            pred_aortic = classifier_aortic.compute_prediction(cvImage)

            return Response({
                    "code": 200,
                    "input_valid": "yes",
                    "pred_pneumonia": pred_pneumonia,
                    "pred_aortic": pred_aortic,
                    "path": path_return
                    })
            
        except MultiValueDictKeyError:
            return Response({
                "code":400,
                "message":"No image selected!"
            })
        except TypeError:
            return Response({
                "code":415,
                "message": "File type not allowed! You must upload images type: jpg, png, jpeg,..."
            })
        except:
            return Response({
                "code":500,
                "message":"Smt bad has been occured!"
            })

