from flask import Flask, request, send_file
import csv
import os
from os.path import join, dirname, realpath
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

def knn(filepath, operation):

    # Importing the dataset
    dataset = pd.read_csv(filepath)

    # Set pre-processign type
    if(operation == 'dropnarows'):
        dataset = dataset.dropna()
    elif(operation == 'dropnacols'):
        dataset = dataset.dropna(axis='columns')
    elif(operation == 'replacenan'):
        dataset = dataset.mean()

    X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
    y = dataset.iloc[:, 0].values

    # Fitting classifier to the Training set   
    classifier = KNeighborsClassifier(n_neighbors=2)
    classifier.fit(X, y)

    # Predict
    y_pred = classifier.predict(X)

    # Write prediction in new CSV file
    predFilePath = UPLOAD_FOLDER + "\\pred.csv"
    f = open(predFilePath, 'a')
    #f.write('PredictedVal' + '\n')
    for val in y_pred:
        f.write(str(val) + '\n')
    f.close()

    return predFilePath 

def svm(filepath, operation):

    # Importing the dataset
    dataset = pd.read_csv(filepath)

    # Set pre-processign type
    if(operation == 'dropnarows'):
        dataset = dataset.dropna()
    elif(operation == 'dropnacols'):
        dataset = dataset.dropna(axis='columns')
    elif(operation == 'replacenan'):
        dataset = dataset.mean()

    X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
    y = dataset.iloc[:, 0].values

    # Fitting classifier to the Training set   
    classifier = SVC(kernel = 'rbf', C = 10.0, gamma = 1.0)
    classifier.fit(X, y)
    y_pred = classifier.predict(X)

    # Write prediction in new CSV file
    predFilePath = UPLOAD_FOLDER + "\\pred.csv"
    f = open(predFilePath, 'a')
    #f.write('PredictedVal' + '\n')
    for val in y_pred:
        f.write(str(val) + '\n')
    f.close()

    return predFilePath 

def gnb(filepath, operation):

    # Importing the dataset
    dataset = pd.read_csv(filepath)

    # Set pre-processign type
    if(operation == 'dropnarows'):
        dataset = dataset.dropna()
    elif(operation == 'dropnacols'):
        dataset = dataset.dropna(axis='columns')
    elif(operation == 'replacenan'):
        dataset = dataset.mean()

    X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
    y = dataset.iloc[:, 0].values

    # Fitting classifier to the Training set   
    gnb = GaussianNB()
    y_pred = gnb.fit(X, y).predict(X)

    # Write prediction in new CSV file
    predFilePath = UPLOAD_FOLDER + "\\pred.csv"
    f = open(predFilePath, 'a')
    #f.write('PredictedVal' + '\n')
    for val in y_pred:
        f.write(str(val) + '\n')
    f.close()

    return predFilePath 

# Upload folder
UPLOAD_FOLDER = 'static\\files'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Get the uploaded files
@app.route("/<algorithm>/<operation>", methods=['POST'])
def uploadFiles(algorithm, operation):
    # get the uploaded file
    uploaded_file = request.files['content']
    if uploaded_file.filename != '':
        # set the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # save the file
        uploaded_file.save(file_path)

        if(algorithm == "knn"):
            pred_path = knn(file_path, operation)
        elif(operation == 'svm'):
            pred_path = svm(file_path, operation)
        elif(operation == 'gnb'):
            pred_path = gnb(file_path, operation) 
    
    return send_file(pred_path, as_attachment=True, download_name='pred.csv')
    
@app.get('/shutdown')
def shutdown():
    shutdown_server()
    return 'Server shutting down...'

def shutdown_server():
    func = request.environ.get('werkzeug.server.shutdown')
    if func is None:
        raise RuntimeError('Not running with the Werkzeug Server')
    func()

if (__name__ == "__main__"):
    app.run(port = 5000)

