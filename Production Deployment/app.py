from flask import Flask

import pickle
from flask import request, jsonify

app = Flask(__name__)

gender_map = {"F": 0, "M": 1}
bp_map = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
cholesterol_map = {"HIGH": 0, "NORMAL": 1}
drug_map = {0: "DrugY", 3: "drugC", 4: "drugX", 1: "drugA", 2: "drugB"}

def predict_drug(Age, 
                 Sex, 
                 BP, 
                 Cholesterol, 
                 Na_to_K):

    # 1. Read the machine learning model from its saved state ...
    pickle_file = open('model.pkl', 'rb')     
    model = pickle.load(pickle_file)
    
    # 2. Transform the "raw data" passed into the function to the encoded / numerical values using the maps / dictionaries
    Sex = gender_map[Sex]
    BP = bp_map[BP]
    Cholesterol = cholesterol_map[Cholesterol]

    # 3. Make an individual prediction for this set of data
    y_predict = model.predict([[Age, Sex, BP, Cholesterol, Na_to_K]])[0]

    # 4. Return the "raw" version of the prediction i.e. the actual name of the drug rather than the numerical encoded version
    return drug_map[y_predict]  

@app.route("/")
def hello():
    return "A test web service for accessing a machine learning model to make drug recommendations v2."

@app.route('/drug', methods=['GET'])
def api_all():
#    return jsonify(data_science_books)

    Age = int(request.args['Age'])
    Sex = request.args['Sex']
    BP = request.args['BP']
    Cholesterol = request.args['Cholesterol']
    Na_to_K = float(request.args['Na_to_K'])

    drug = predict_drug(Age, Sex, BP, Cholesterol, Na_to_K)

    #return(jsonify(drug))
    return(jsonify(recommended_drug = drug))

