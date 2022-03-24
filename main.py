from flask import Flask, request
import os
from os.path import join, dirname, realpath
import pandas as pd

app = Flask(__name__)

# enable debugging mode
app.config["DEBUG"] = True

# Upload folder
UPLOAD_FOLDER = 'static/files'
app.config['UPLOAD_FOLDER'] =  UPLOAD_FOLDER

# Get the uploaded files
@app.route("/<fileName>/<operation>", methods=['POST'])
def uploadFiles():
    # get the uploaded file
    uploaded_file = request.files['file']
    if uploaded_file.filename != '':
        # set the file path
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], uploaded_file.filename)
        # save the file
        uploaded_file.save(file_path)
        ####################
        #
        # clustering script call
        #
        ####################
        # True for now
    return True

if (__name__ == "__main__"):
     app.run(port = 5000)
