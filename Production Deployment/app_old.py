# https://stackoverflow.com/questions/54063285/numpy-is-already-installed-with-anaconda-but-i-get-an-importerror-dll-load-fail
# Don't forget to launch VS Code by typing "code" into the terminal window

# https://programminghistorian.org/en/lessons/creating-apis-with-python-and-flask
# https://code.visualstudio.com/docs/python/tutorial-flask

# http://127.0.0.1:5000/drug?Age=47&Sex=F&BP=LOW&Cholesterol=HIGH&Na_to_K=14
# http://127.0.0.1:5000/drug?Age=60&Sex=F&BP=LOW&Cholesterol=HIGH&Na_to_K=20

# https://graham-harrison68-web02.azurewebsites.net/drug?Age=47&Sex=F&BP=LOW&Cholesterol=HIGH&Na_to_K=14

# Old code that used to be included in launch.json. If you launch VS Code from anaconda it is not necessary
# {
#     "python.condaPath": "C:\\Users\\GHarrison\\Anaconda3",
#     "version": "0.2.0",


import flask
import pickle
from flask import request, jsonify

app = flask.Flask(__name__)
app.config["DEBUG"] = True

gender_map = {"F": 0, "M": 1}
bp_map = {"HIGH": 0, "LOW": 1, "NORMAL": 2}
cholestol_map = {"HIGH": 0, "NORMAL": 1}
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
    Cholesterol = cholestol_map[Cholesterol]

    # 3. Make an individual prediction for this set of data
    y_predict = model.predict([[Age, Sex, BP, Cholesterol, Na_to_K]])[0]

    # 4. Return the "raw" version of the prediction i.e. the actual name of the drug rather than the numerical encoded version
    return drug_map[y_predict]  

@app.route('/', methods=['GET'])
def home():
    return '''<h1>Drug Recommendation Web Service</h1>
<p>A test web service for accessing a machine learning model to make drug recommendations v3.</p>'''

# A route to return all of the available entries in our catalog.
@app.route('/drug', methods=['GET'])
def api_all():
#    return jsonify(data_science_books)

    Age = int(request.args['Age'])
    Sex = request.args['Sex']
    BP = request.args['BP']
    Cholesterol = request.args['Cholesterol']
    Na_to_K = float(request.args['Na_to_K'])

    drug = predict_drug(Age, Sex, BP, Cholesterol, Na_to_K)

    return(jsonify(recommended_drug = drug))
    #return(jsonify(drug))

app.run()