#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Libraries
import os
import sys
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# for graphical out of df.head and df.describe from within a function
from IPython.display import display

# Feature Engineering
from feature_engine.categorical_encoders import OneHotCategoricalEncoder

# Model and Metrics
from sklearn.linear_model import LinearRegression


# In[2]:


# Run environment Setup
# global variables available to all functions in this python file
TRAINED_MODEL = 0
OHCE = 0
TO_DROP = []


# #### Read Data

# In[3]:


def read_data(filename):
    print("\n***** FUNCTION read_data*****")
    df = pd.read_csv(os.path.join(application.config['UPLOAD_FOLDER'],filename))

    # See the data in the df
    display(df.head())

    # Full Data set Shape
    print(df.shape)
    
    return(df)
# end of function read_data


# #### Data Exploration

# In[4]:


def disp_df_info(df):
    print("\n*****FUCNTION disp_df_info*****")
    # Correlation visualized 
    sns.pairplot(df)
    plt.show()
    
    print("\n Column Headers:")
    print(df.columns)

    # print first 10 data samples
    print("Top 10 rows:")
    display(df.head())
    
    # Describe the df
    print("\nStatistics:")
    display(df.describe())

    # Identify the Categorical Vars and identify nulls
    print("\nInformation:")
    print(df.info())

    # Count Nulls
    print("\nNull Count:")
    print(df.isnull().sum())

    # Percent of Nulls
    print("\nNull Percent:")
    print(df.isnull().mean())
    
# end of function disp_df_info


# #### Feature Engineering

# In[5]:


def feature_engineering(df_input):
    print("\*****FUNCTION feature_engineering*****")

    df = df_input.copy(deep=True)
    global OHCE

    # FE on 5 columns
    # Convert the categorical vars to k-1 dummy vars
    OHCE= OneHotCategoricalEncoder(variables=['State'], # we can select which variables to encode
                                   drop_last=True)      # to return k-1, false to return k


    OHCE.fit(df)
    df = OHCE.transform(df)
    print(OHCE.encoder_dict_)
    
    return(df)
# end of feature_engineering function


# #### Feature Selection

# In[6]:


def feature_selection(df_input):
    print("\n*****FUCNTION feature_selction****")

    df = df_input.copy(deep=True)
    
    global TO_DROP

    # Check the corr of the variables after encoding
    corr_mat = df.corr()
    print(corr_mat)

    # Correlation Matrix visualized as HeatMap
    plt.figure(figsize=(6,6))
    sns.heatmap(corr_mat, annot=True, cmap='coolwarm', center = 0 , vmin=-1, vmax=1)
    plt.show()

    # Shape before dropping features
    print('Shape BEFORE Dropping features:', df.shape)
    
    # Drop features where the correlation is less than .5 with the target / Y var
    TO_DROP = ['Administration','State_New York', 'State_California']
    
    df.drop(df[TO_DROP], axis= 1 , inplace = True)
    display(df.head(5))
    
    # Shape after dropping features
    print('Shape AFTER Dropping features:', df.shape)
    
    return(df)
# end of feature_selection function


# #### Data Split into X/Feature and Y/target

# In[7]:


def data_split(df_input):
    print("\n*****FUNCTION data_split*****")
    
    df = df_input.copy(deep=True)
    
    # Create Y var
    y = df['Profit']
    display(y.head())

    # Create X Vars
    x = df.drop(['Profit'], axis = 1)
    display(x.head())
    
    return(x,y)
# end of function data_split


# #### Feature Scaling

# In[8]:


#### sklearn LinearRegression() performs scaling


# #### Model Fitting 1 -Keep R&D Spend and Marketing Spend<br>Based on the correlation Matrix, keep ONLY X vars where linear correlation with Y var is greater than 0.5

# In[9]:


def build_linreg_model(x_input,y_input):
    print("\n*****FUNCTION build_linreg_model*****")

    x = x_input.copy(deep=True)
    y = y_input.copy(deep=True)

    # Call Linear Regression
    mod1 = LinearRegression()
    mod1 = LinearRegression().fit(x, y)

    # Print the Intercept and the coef
    print('intercept:', mod1.intercept_)
    print('Coefficients:', mod1.coef_)
    
    # Predict using the model
    y_pred = mod1.predict(x)
    print(y_pred)
       

    #### Model Metrics - 1

    # Score the model
    r_sq = mod1.score(x, y)
    print('coefficient of determination/R squared:', r_sq)

    # Calculate the Adj R2
    n = x.shape[0] # number of observations
    p = x.shape[1] # number of x vars
    print('Number of Observations:',n)
    print('Number of Regressors:', p)
    adj_r2 = 1 - (1 - r_sq ) * ((n - 1) / (n -p - 1))
    print('Adjusted R squared:', adj_r2)
    
    # Print Completion
    print('***********************Model Ready to be used/invoked*************************')

    return(mod1,r_sq)
# end of build_linreg_model function


# #### Create App

# In[10]:


# Import Flask 
from flask import Flask
from flask import render_template
from flask import request
# from flask import send_file


# In[11]:


# import werkzeug to run your app as a web application
# from werkzeug.serving import run_simple


# In[12]:


# Create input file folder
upload_folder_name = 'input_startup_folder'
upload_folder_path = os.path.join(os.getcwd(),upload_folder_name)
print('Upload folder path is:',upload_folder_path)
if not os.path.exists(upload_folder_path):
    os.mkdir(upload_folder_path)


# In[13]:


# Instantiate the app using Flask
application = Flask(__name__)


# In[14]:


application.config['UPLOAD_FOLDER'] = upload_folder_path


# In[15]:


# home displays trainform.html
@application.route("/train", methods=['GET'])
def train():
    return render_template('trainform.html')
# end of home


# In[16]:


# submit on trainform.html
@application.route("/build_mod", methods=['POST'])
def build_mod():
    
    global TRAINED_MODEL
    
    file_obj = request.files.get('traindata')
    print("Type of the file is :", type(file_obj))
    name = file_obj.filename
    print(name)
    file_obj.save(os.path.join(application.config['UPLOAD_FOLDER'],name))
    
    # Is the File extension .csv
    if name.lower().endswith('.csv'):
        print('Input File extension good', name)
    else:
        print('***ERROR*** Input file extension NOT good')
        return render_template('trainform.html', errstr = "***ERROR*** Input file extension NOT good") 
    #End If
        
    startup_df = read_data(name)
    disp_df_info(startup_df)
    eng_df = feature_engineering(startup_df)
    sel_df = feature_selection(eng_df)
    x,y = data_split(sel_df)
    TRAINED_MODEL,score=build_linreg_model(x,y)

    return render_template('trainresults.html',acc=score)
# end of home


# In[17]:


# Use model on trainresults.html
# OR Use model on predresults.html
@application.route("/use", methods=['POST','GET'])
def use():
    return render_template('predform.html')
# end of home


# In[18]:


# submit on predform.html
@application.route("/make_pred", methods=['POST'])
def make_pred():
    req = request.form
    print(req)              # req is a ImmutableMultiDict
    newdata = []
    for k, v in req.items():
        newdata.append(v)
    newdata = [newdata]
    new_df = pd.DataFrame(data=newdata,     columns = ['Profit', 'R&D Spend', 'Administration', 'Marketing Spend', 'State'])  
    print('New Data:')
    display(new_df)

    # Feature Eng - Reuse the OHCE
    print(OHCE.encoder_dict_)
    neweng_df = OHCE.transform(new_df)
    print('New FE Data:')
    display(neweng_df)

    # Feature selection - Reuse TO_DROP
    newsel_df = neweng_df.drop(neweng_df[TO_DROP], axis=1)
    print('New Selected Data:')
    display(newsel_df)

    #Split Data for the 1 record entered thru the form
    new_x,new_y = data_split(newsel_df)

    # Make Prediction - Reuse MODEL to make prediction
    new_pred = TRAINED_MODEL.predict(new_x)
    print('\nNew Prediction:',new_pred)
  
    print("\n*********************** New Prediction Complete WITH FLASK **************************")

    # Return results to browser/client as a df and prediction 
    return render_template('predresults.html',data=new_df, pred=new_pred)  

# end of make_pred


# #### Main Program for Web App

# In[ ]:


# Main Program for Web App
# If __name__ = __main__, program is running standalone
if __name__ == "__main__":
    print("Python script is run standalone")
    print("Python special variable __name__ =", __name__)  
    
 
    # Run the flask app in jupyter noetbook needs run_simple 
    # Run the flask app in python script needs app.run
#     run_simple('172.17.0.10', 5000, app, use_debugger=True)
    application.run('0.0.0.0',debug=True)

else:
    # __name__ will have the name of the module that imported this script
    print("Python script was imported")
    print("Python special variable __name__ =", __name__)   
#End Main program

