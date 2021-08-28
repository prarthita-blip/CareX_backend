from flask import Flask,request
import joblib
app = Flask(__name__)

@app.route("/")
def hello():
    return "Server is Up and Running!"

@app.route('/diabetes_predict', methods=['POST'])
def diabetes():
    data= request.json
    values=[]
    print(data)
    for i in data:
        values.append(data[i])
    clf=joblib.load('Diabetes_Predictor.pkl')
    prediction=clf.predict([values])[0]
    return str(prediction)

@app.route('/liver_predict', methods=['POST'])
def liver():
    data= request.json
    values=[]
    print(data)
    for i in data:
        values.append(data[i])
    clf=joblib.load('liver_Predictor.pkl')
    prediction=clf.predict([values])[0]
    return str(prediction)

@app.route('/heartAttack_predict', methods=['POST'])
def heartAttack():
    data= request.json
    values=[]
    print(data)
    for i in data:
        values.append(data[i])
    clf=joblib.load('HeartAttack_Predictor.pkl')
    prediction=clf.predict([values])[0]
    return str(prediction)

@app.route('/kidney_predict', methods=['POST'])
def Kidney():
    data= request.json
    values=[]
    print(data)
    for i in data:
        values.append(data[i])
    clf=joblib.load('Kidney_Predictor.pkl')
    prediction=clf.predict([values])[0]
    return str(prediction)

@app.route('/animia_predict', methods=['POST'])
def Anemia():
    data= request.json
    values=[]
    print(data)
    for i in data:
        values.append(data[i])
    clf=joblib.load('Anemia_Predictor.pkl')
    prediction=clf.predict([values])[0]
    return str(prediction)
if __name__ == '__main__':
    app.run(debug=True)