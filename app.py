from flask import Flask,jsonify, request
import response
# functions from make_ python file((
from make_prediction import get_data, preprocessing, get_predictions

app = Flask(__name__)
@app.route('/predict_churn', methods=['POST', 'GET'])

def make_pred():
    df = get_data()
    df['Churn_Prediction'] = get_predictions(preprocessing(get_data()))
    df = df[df['Churn_Prediction'] == 1]
    results = df[['CustomerID', 'Churn_Prediction']].to_dict(orient='records')
    return jsonify(results)

if __name__ == '__main__':
    app.run(port=8080,debug=True)