
from flask_sqlalchemy import SQLAlchemy
from flask import Flask, render_template, flash, request, jsonify
from database import db
from flask_cors import CORS, cross_origin

DEBUG = True
app = Flask(__name__)
app.config.from_object(__name__)
app.config.from_object('config.DevelopmentConfig')#os.environ['APP_SETTINGS'])
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
CORS(app)

db.init_app(app)

from classe import Historico, Users
import json
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import uuid
import os
import requests
import pickle as serializer
from werkzeug.security import generate_password_hash, check_password_hash
import helper

class DenseNet121(nn.Module):
    """Model modified.
    The architecture of our model is the same as standard DenseNet121
    except the classifier layer which has an additional sigmoid function.
    """
    def __init__(self, out_size):
        super(DenseNet121, self).__init__()
        self.densenet121 = torchvision.models.densenet121(pretrained=True)
        num_ftrs = self.densenet121.classifier.in_features
        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_ftrs, out_size),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.densenet121(x)
        return x

class HeatmapGenerator ():
    
    #---- Initialize heatmap generator
    #---- pathModel - path to the trained densenet model
    #---- nnArchitecture - architecture name DENSE-NET121, DENSE-NET169, DENSE-NET201
    #---- nnClassCount - class count, 14 for chxray-14

 
    def __init__ (self, pathModel, nnClassCount, transCrop):
       
        #---- Initialize the network
        model = DenseNet121(nnClassCount)#.cuda()
        
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model).cuda()
        else:
            model = torch.nn.DataParallel(model)
        
        modelCheckpoint = torch.load(pathModel, map_location=torch.device('cpu'))
        model.load_state_dict(modelCheckpoint['state_dict'])

        self.model = model
        self.model.eval()
        
        #---- Initialize the weights
        self.weights = list(self.model.module.densenet121.features.parameters())[-2]

        #---- Initialize the image transform
        normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        transformList = []
        transformList.append(transforms.Resize((transCrop, transCrop)))
        transformList.append(transforms.ToTensor())
        transformList.append(normalize)  
        self.transformSequence = transforms.Compose(transformList)
    
    #--------------------------------------------------------------------------------
     
    def generate (self, pathImageFile, pathOutputFile, transCrop):
        class_names = ['Normal', 'Cardiomediastino Alargado', 'Cardiomegalia', 'Opacidade pulmonar', 
                'Lesao de Pulmao', 'Edema', 'Consolidaçao', 'Pneumonia', 'Atelectasia', 'Pneumotorax', 
                'Derrame pleural', 'Outro pleural', 'Fratura', 'Dispositivos de Suporte']
        #---- Load image, transform, convert 
        with torch.no_grad():
            imageData = Image.open(pathImageFile).convert('RGB')
            imageData = self.transformSequence(imageData)
            imageData = imageData.unsqueeze_(0)
            if torch.cuda.is_available():
                imageData = imageData.cuda()
            l = self.model(imageData)
            output = self.model.module.densenet121.features(imageData)
            probs = []
            for probabilities in l.tolist():
              for prob in range(0,len(probabilities)):
                probabilidade = {}
                probabilidade['porcentagem'] = round(probabilities[prob]*100,2)
                probabilidade['classe'] = class_names[prob]
                probs.append(probabilidade)
            #---- Generate heatmap
            heatmap = None
            for i in range (0, len(self.weights)):
                map = output[0,i,:,:]
                if i == 0: heatmap = self.weights[i] * map
                else: heatmap += self.weights[i] * map
                npHeatmap = heatmap.cpu().data.numpy()

        #---- Blend original and heatmap 
                
        imgOriginal = cv2.imread(pathImageFile, 1)
        imgOriginal = cv2.resize(imgOriginal, (transCrop, transCrop))
        
        cam = npHeatmap / np.max(npHeatmap)
        cam = cv2.resize(cam, (transCrop, transCrop))
        heatmap = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
        
        img = cv2.addWeighted(imgOriginal,1,heatmap,0.35,0)            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        cv2.imwrite(pathOutputFile, img)
        
        return probs

         

@app.route("/api/predict", methods=['POST'])
@cross_origin()
@helper.token_required
def api(current_user):
    f = request.files['exame']
    code = str(uuid.uuid1())
    f.save('static/'+code+f.filename)
    
    pathInputImage = 'static/' + code + f.filename
    pathOutputImage = 'static/h-' + code + f.filename

    pathModel = "model_zeroes_1epoch_densenet.pth.tar"

    nnClassCount = 14
    imgtransCrop = 224

    h = HeatmapGenerator(pathModel, nnClassCount, imgtransCrop)
    
    retorno = {}
    retorno['predicao'] = h.generate(pathInputImage, pathOutputImage, imgtransCrop)
    retorno['heatmap'] = 'https://tcc-guivm.herokuapp.com/'+pathOutputImage
    retorno['image'] = 'https://tcc-guivm.herokuapp.com/'+pathInputImage
    retorno['nomePac'] = request.form.get('nome')
    retorno['cpfPac'] = request.form.get('cpf')
    retorno['tipoExame'] = request.form.get('tipoExame')
    retorno['nomeMedico'] = current_user.name

    historico = Historico(name=request.form.get('nome'),cpf= request.form.get('cpf'),tipoExame = request.form.get('tipoExame'), response = str(retorno))
    db.session.add(historico)
    db.session.commit()

    return app.response_class(
        response = json.dumps(retorno,indent=True),
        status=200,
        mimetype='application/json'
    )

@app.route("/api/list")
@cross_origin()
@helper.token_required
def get_all(current_user):
    try:
        historicos = Historico.query.all()
        return  jsonify([e.serialize() for e in historicos])
    except Exception as e:
	    return(str(e))

@app.route("/api/user/login", methods=['POST'])
@cross_origin()
def login():
    try:
        return helper.auth()
    except Exception as e:
	    return(str(e))

@app.route("/api/user/create", methods=['POST'])
@cross_origin()
def userCreate():
    user = Users(name= request.form.get('nome'), cpf= request.form.get('cpf'), crm = request.form.get('crm'), email = request.form.get('email'), password= generate_password_hash(request.form.get('password')))
    db.session.add(user)
    db.session.commit()

    return app.response_class(
        response = json.dumps({"status":"OK"},indent=True),
        status=200,
        mimetype='application/json'
    )

if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port= port)
