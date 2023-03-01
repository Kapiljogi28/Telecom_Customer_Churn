import pandas as pd
import numpy as np
import os
import warnings
warnings.filterwarnings('ignore')
import datetime
import pickle as pk
import json
from sklearn.preprocessing import LabelEncoder
import mysql.connector

def get_data():
    mydb = mysql.connector.connect(
        host="localhost",
        user="root",
        password="56664444Kj",
        database="project")

    mycursor = mydb.cursor()
    mycursor.execute("SELECT * FROM customer_churn")
    myresult = mycursor.fetchall()
    
    df = pd.DataFrame(myresult,columns=['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip_Code',
    'Lat_Long', 'Latitude', 'Longitude', 'Gender', 'Senior_Citizen',
    'Partner', 'Dependents', 'Tenure_Months', 'Phone_Service',
    'Multiple_Lines', 'Internet_Service', 'Online_Security',
    'Online_Backup', 'Device_Protection', 'Tech_Support', 'Streaming_TV',
    'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method',
    'Monthly_Charges', 'Total_Charges', 'Churthn_Label', 'Churn_Value',
    'Churn_Score', 'CLTV', 'Churn_Reason'])
    
    churn_df = df.drop(['Churthn_Label', 'Churn_Value',
    'Churn_Score', 'CLTV', 'Churn_Reason','Count', 'Country', 'State', 'City', 'Zip_Code',
    'Lat_Long', 'Latitude', 'Longitude'],axis=1)
    
    return churn_df
    
def preprocessing(churn_df):
    
    # Total charges are in object dtype so convert into Numerical feature 
    churn_df['Total_Charges'] = pd.to_numeric(churn_df['Total_Charges'], errors='coerce')
    
    # replace NaN values with mean value
    churn_df.Total_Charges = churn_df.Total_Charges.fillna(churn_df.Total_Charges.median())
    
    churn_df['Gender'] = churn_df['Gender'].replace({'Male': 1, 'Female': 0})

    y_n_cols = ['Senior_Citizen', 'Partner', 'Dependents', 'Phone_Service', 'Multiple_Lines','Online_Security', 'Online_Backup', 
        'Device_Protection', 'Tech_Support', 'Streaming_TV', 'Streaming_Movies', 'Paperless_Billing']
    churn_df[y_n_cols] = churn_df[y_n_cols].replace({'Yes':1, 'No':0}).replace(regex=r'No.*', value=0)

    churn_df['Internet_Service'] = churn_df['Internet_Service'].replace({'Fiber optic': 0, 'DSL': 1, 'No': 2})
    churn_df['Contract'] = churn_df['Contract'].replace({'Month-to-month': 0, 'Two year': 1, 'One year': 2})
    churn_df['Payment_Method'] = churn_df['Payment_Method'].replace({'Electronic check': 0,
    'Mailed check': 1,
    'Bank transfer (automatic)': 2,
    'Credit card (automatic)': 3})
    
    encoder = LabelEncoder()
    churn_df['CustomerID'] = encoder.fit_transform(churn_df['CustomerID'])

    print('**'*50)
    # print(churn_df['CustomerID'].unique())
    return churn_df

def get_predictions(churn_df):
    model = pk.load(open('C:/Users/KAPIL SANTOSH JOGI/Downloads/New folder/Telecom Customer Churn Project/Model_20230223/model.pkl', 'rb'))
    
    return model.predict(churn_df)
    

def main():
    churn_df = get_data()
    churn_df['Churn_Prediction'] = get_predictions(preprocessing(get_data()))

    churn_df  = churn_df[churn_df['Churn_Prediction'] == 1]
    results = churn_df[['CustomerID', 'Churn_Prediction']].to_dict(orient='records')

    path = 'C:/Users/KAPIL SANTOSH JOGI/Downloads/New folder/Telecom Customer Churn Project/Predictions'
    try:
        os.mkdir(path)
        with open(f"{path}/Predictions_{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')}.json", "w") as f:
            json.dump(results, f)
    except:
        with open(f"{path}/Predictions_{datetime.datetime.strftime(datetime.datetime.now(), '%Y%m%d')}.json", "w") as f:
            json.dump(results, f)

if __name__ == '__main__':
    get_data()
    get_predictions(preprocessing(get_data()))
    main()
