import os
import sys
from src.exception import CustomException               # src folder sey exception manga liya
from src.logger import logging                          # logginh bhi
import pandas as pd                                 
from sklearn.model_selection import train_test_split    # data leney pe split karke transformation ke liye bejengay
from dataclasses import dataclass                       # used to create class variable

from src.components.data_transfrormation import DataTransformation
from src.components.data_transfrormation import DataTransformConfig

from src.components.model_trainer import ModelTrainerConfig
from src.components.model_trainer import ModelTrainer

''' ARTIFACTS KO GIT IGNORE MAI DALNA WHEN U WILL MAKE FINAL GIT REPOSITORY 
'''

# input chaiye ingestion ke liye
# sarey types of inputs diff diff classes mai rakengey

@dataclass                      # decorator yt karo # without init use kar app varibale define ka rhe ho
class DataIngestionConfig :     # any required input will be assembled in this class
    train_data_path: str=os.path.join('artifacts',"train.csv")  #artifact folder train.csv file and these two will be stored in train_data path(basically they wil have the path)
                                                                # this is with respect to training data path
    test_data_path: str=os.path.join('artifacts',"test.csv")
    raw_data_path: str=os.path.join('artifacts',"raw.csv")

# @dataclass  (dontuse)     # if u have to define functions in the class then use that init constructor
class DataIngestion:    
    def __init__(self) :
        self.ingestion_config=DataIngestionConfig() #object creation in diffrent class
                                                    #when we call class DataIngetion tabhi uper 3 ke variables of class 
                                                    #DAtaigestionconfig sidha (class variable)ingestion config mai 
                                                    #akey store ho jayega
    def initiate_data_ingestion(self):      #to read code from data base # jasie utils mai sql aur mongb ka code yaha manga ka we can access it
        logging.info("enterd the data ingestion method or component")
        try:
            df=pd.read_csv('data\\stud.csv')         # easy method of reading    #use mongofb or sql here
            logging.info('read the data set as dataframe')  # keep logging in the data
 
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True) #creats new directory(folder) by makedir and os.path.dir
                                                                                            #name sey directory kaa name obtain karta hai 
                                                                                            #then hamne jo upper class ka object banaya hai 
                                                                                            #waha sey eak varibale(folder,filename) call karte ha
                                                                                            #and phir 

            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)  #raw data path stored in artifact folder

            logging.info("train test split initiated")
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True) # yaha bhi same train data is stores in artifact folder
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True) # yaha bhi same test data is stores in artifact folder
            logging.info("ingestion of data is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path,   # gettting my data ready for transformation so uskey liye return generate kiya hai
            )
        except Exception as e:
            raise CustomException(e,sys)

if __name__=="__main__":
    obj = DataIngestion()           # creat object of the following class
    train_data,test_data = obj.initiate_data_ingestion()     # obj ke andar jo method hai uska return values karo

    data_transformation = DataTransformation()      # object creat wo dusri file sey yaha import  kiya hai
    train_arr,test_arr,preprocess =data_transformation.initiate_data_transformation(train_data,test_data)      # perform 

    model_trainer=ModelTrainer()
    score,model = model_trainer.initiate_model_trainer(train_arr,test_arr)
    print(score,model)