import json
import pickle

from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

# Create the application.
app=Flask(__name__)
## Load the model
regmodel=pickle.load(open('regmodel.pkl','rb')) # load the regression model
scalar=pickle.load(open('scaling.pkl','rb')) # load the scalar model

# Home page ROOT
@app.route('/')
def home():
    return render_template('home2.html')

# create a predict api 
@app.route('/predict_api',methods=['POST']) # POST request
def predict_api():
    data=request.json['data'] # transform the data into json format
    print(data) # print the data
    print(np.array(list(data.values())).reshape(1,-1)) # transform the data into array format
    new_data=scalar.transform(np.array(list(data.values())).reshape(1,-1)) # scale the data
    output=regmodel.predict(new_data) # predict the data
    print(output[0]) # print the output
    return jsonify(output[0]) # return the output



# @app.route('/predict',methods=['POST'])
# def predict():
#     data=[float(x) for x in request.form.values()]
#     final_input=scalar.transform(np.array(data).reshape(1,-1))
#     print(final_input)
#     output=regmodel.predict(final_input)[0]
#     return render_template("home.html",prediction_text="The House price prediction is {}".format(output))



if __name__=="__main__":
    app.run(debug=True)
   
     
