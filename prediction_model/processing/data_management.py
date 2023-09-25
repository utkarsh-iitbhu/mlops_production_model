# Import libraries
import os 
import pandas as pd
import joblib

# Import other files/modules 
from prediction_model.config import config
# Imports all the LOCAL and GLOBAL paths and variables 

def load_dataset(file_name):
    '''Read data'''
    file_path = os.path.join(config.DATAPATH,file_name) # DATAPATH is dataset dir 
    _data = pd.read_csv(file_path)
    return _data 

def save_pipeline(pipeline_to_save):
    """ Store output of pipeline 
        Exporting pickle file of trained model
    """
    save_file_name = 'classification_v1.pkl'
    save_path = os.path.join(config.SAVED_MODEL_PATH,save_file_name)
    joblib.dump(pipeline_to_save,save_path)
    print("Saved pipeline :",save_file_name)
    
def load_pipeline(pipeline_to_load):
    '''Importing the pickle file'''
    save_path = os.path.join(config.SAVED_MODEL_PATH,pipeline_to_load)
    trained_model = joblib.load(save_path)
    return trained_model
    
    
    
