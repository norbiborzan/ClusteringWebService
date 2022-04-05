from flask import Flask, request, send_file
import pandas as pd
import csv
import os
from os.path import join, dirname, realpath
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

app = Flask(__name__)

# Enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static\\files'
PRED_PATH = 'static\\files\\pred.csv'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def predict(filepath, algorithm, operation):

    # Importing the dataset
    dataset = pd.read_csv(filepath)

    # Set pre-processign type
    if(operation == 'dropnarows'):
        dataset = dataset.dropna()
    elif(operation == 'dropnacols'):
        dataset = dataset.dropna(axis='columns')
    elif(operation == 'replacenan'):
        dataset = dataset.mean()

    # Choose columns
    X = dataset.iloc[:, [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]].values
    y = dataset.iloc[:, 0].values

    if(algorithm == 'knn'):
        # Fitting classifier to the dataset   
        classifier = KNeighborsClassifier(n_neighbors=2)
        classifier.fit(X, y)
        # Predict
        y_pred = classifier.predict(X)
    elif(algorithm == 'svm'):
        # Fitting classifier to the dataset   
        classifier = SVC(kernel = 'rbf', C = 10.0, gamma = 1.0)
        classifier.fit(X, y)
        # Predict
        y_pred = classifier.predict(X)
    elif(algorithm == 'gnb'):
        # Fitting classifier to the dataset   
        gnb = GaussianNB()
        # Predict
        y_pred = gnb.fit(X, y).predict(X)
    elif(algorithm == 'compare'):
        #KNN
        classifier = KNeighborsClassifier(n_neighbors=2)
        classifier.fit(X, y)
        y_predKNN = classifier.predict(X)
        #SVM
        classifier = SVC(kernel = 'rbf', C = 10.0, gamma = 1.0)
        classifier.fit(X, y)
        y_predSVM = classifier.predict(X)
        #GNB
        gnb = GaussianNB()
        y_predGNB = gnb.fit(X, y).predict(X)
    
    if(algorithm == 'compare'):
        dataset.insert(0, "KNN Predicted Class", y_predKNN, True)
        dataset.insert(1, "SVM Predicted Class", y_predSVM, True)
        dataset.insert(2, "GNB Predicted Class", y_predGNB, True)
        dataset.to_csv(PRED_PATH, index=False)
    else:
        dataset.insert(0, "Predicted Class", y_pred, True)
        dataset.to_csv(PRED_PATH, index=False)

# Get the uploaded files
@app.route("/<algorithm>/<operation>", methods=['POST'])
def uploadFiles(algorithm, operation):

    # Det the uploaded file
    uploaded_file = request.files['content']
    if uploaded_file.filename != '':
        # Set the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # Remove the dataset file if it exists
        if os.path.exists(file_path):
            os.remove(file_path)
        # Remove the predicted file if it exists
        if os.path.exists(os.path.join(app.config['UPLOAD_FOLDER'], "pred.csv")):
            os.remove(os.path.join(app.config['UPLOAD_FOLDER'], "pred.csv"))
        # Save the file
        uploaded_file.save(file_path)
        # Run the prediction algorithms
        predict(file_path, algorithm, operation) 
    # Return predicted file to client
    return send_file(PRED_PATH, as_attachment=True, download_name='pred.csv')

# Shutdown the server 
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

