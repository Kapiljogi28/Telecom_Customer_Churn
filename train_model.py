import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import RandomForestClassifier
from imblearn.combine import SMOTEENN
from collections import Counter
import sqlalchemy as sa
from sqlalchemy import create_engine, text
import pickle
import datetime
import warnings
warnings.filterwarnings('ignore')
import matplotlib.ticker as mtick
import mysql.connector


# Data Gathering & Data Analysis
def get_data():
    
    # Engine=sa.create_engine("mysql+pymysql://root:56664444Kj@localhost:3306/project")
    # churn_df = pd.read_sql_table('telco_customer_churn',Engine)
    mydb = mysql.connector.connect(
    host="localhost",
    user="root",
    password="56664444Kj",
    database="project")

    mycursor = mydb.cursor()

    mycursor.execute("SELECT * FROM customer_churn")

    myresult = mycursor.fetchall()
    churn_df = pd.DataFrame(myresult, columns= ['CustomerID', 'Count', 'Country', 'State', 'City', 'Zip_Code',
    'Lat_Long', 'Latitude', 'Longitude', 'Gender', 'Senior_Citizen',
    'Partner', 'Dependents', 'Tenure_Months', 'Phone_Service',
    'Multiple_Lines', 'Internet_Service', 'Online_Security',
    'Online_Backup', 'Device_Protection', 'Tech_Support', 'Streaming_TV',
    'Streaming_Movies', 'Contract', 'Paperless_Billing', 'Payment_Method',
    'Monthly_Charges', 'Total_Charges', 'Churthn_Label', 'Churn_Value',
    'Churn_Score', 'CLTV', 'Churn_Reason'])
    return churn_df

# Feature Selection
def preprocess_data(churn_df):
    churn_df.drop(['CustomerID','Count', 'Country', 'State', 'City','Zip_Code','Lat_Long','Longitude',
        'Latitude','Churthn_Label','Churn_Score','CLTV','Churn_Reason'],axis=1,inplace=True)

    # Total charges are in object dtype so convert into Numerical feature 
    churn_df['Total_Charges'] = pd.to_numeric(churn_df['Total_Charges'], errors='coerce')

    # replace NaN values with mean value
    churn_df.Total_Charges = churn_df.Total_Charges.fillna(churn_df.Total_Charges.median())

    #Categorical feature
    categorical_feature = {feature for feature in churn_df.columns if churn_df[feature].dtypes == 'O'}
    # print(f'Count of Categorical feature: {len(categorical_feature)}')
    # print(f'Categorical feature are:\n {categorical_feature}')

    encoder = LabelEncoder()
    for feature in categorical_feature:
        churn_df[feature] = encoder.fit_transform(churn_df[feature])
    
    return churn_df

def split_data(churn_df):
    
    X = churn_df.drop(['Churn_Value'], axis=1)
    Y = churn_df['Churn_Value']
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Using SMOTENN Technique
    st=SMOTEENN()
    x_train_st, y_train_st = st.fit_resample(x_train, y_train)
    #splitting the over sampling dataset
    x_train_sap, x_test_sap, y_train_sap, y_test_sap = train_test_split(x_train_st, y_train_st, test_size=0.2)

    return x_train_sap, x_test_sap, y_train_sap, y_test_sap

def train_model(x_train_sap, y_train_sap):
    # print("The number of classes before fit {}".format(Counter(y_train)))
    # print("The number of classes after fit {}".format(Counter(y_train_st)))
    
    # Random forest classifier
    Rfc_sampling = RandomForestClassifier(n_estimators=150,criterion='gini', max_depth=15, min_samples_leaf=10, min_samples_split=6)
    Rfc_sampling.fit(x_train_sap, y_train_sap)
    
    return Rfc_sampling

def store_model(x_train_sap, x_test_sap, y_train_sap, y_test_sap):
    Rfc_sampling = train_model(x_train_sap, y_train_sap)
    path = f'C:/Users/KAPIL SANTOSH JOGI/Downloads/New folder/Telecom Customer Churn Project/Model_{datetime.datetime.strftime(datetime.datetime.now(), "%Y%m%d")}'
    try:
        os.mkdir(path)
        pickle.dump(Rfc_sampling, open(f'{path}/model.pkl', 'wb'))

        with open(f'{path}/Model_evaluation.txt', 'w') as f:
            content = f"""Model : {Rfc_sampling}\n
            {'*'*10} Train Results {'*'*10}\n
            {classification_report(Rfc_sampling.predict(x_train_sap), y_train_sap)}\n\n
            {'*'*10} Test Results {'*'*10}\n
            {classification_report(Rfc_sampling.predict(x_test_sap), y_test_sap)}"""
            f.write(content)
    except:
        pickle.dump(Rfc_sampling, open(f'{path}/model.pkl', 'wb'))

        with open(f'{path}/Model_evaluation.txt', 'w') as f:
            content = f"""Model : {Rfc_sampling}\n
            {'*'*10} Train Results {'*'*10}\n
            {classification_report(Rfc_sampling.predict(x_train_sap), y_train_sap)}\n\n
            {'*'*10} Test Results {'*'*10}\n
            {classification_report(Rfc_sampling.predict(x_test_sap), y_test_sap)}"""
            f.write(content)
            

if __name__ == '__main__':
    x_train_sap, x_test_sap, y_train_sap, y_test_sap = split_data(preprocess_data(get_data()))
    train_model(x_train_sap, y_train_sap)
    print('*#'*50)
    store_model(x_train_sap, x_test_sap, y_train_sap, y_test_sap)
