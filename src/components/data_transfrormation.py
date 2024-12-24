# here we will be writting the code how to change the categorical features into numerical features
# how to do the label encoding or one hot encoding
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder,StandardScaler

from src.exception import CustomException
from src.logger import logging
from sklearn.pipeline import Pipeline
from src.utils import save_object

@dataclass
class DataTransformConfig:
    preprocessor_obj_file_path = os.path.join('artifacts',"preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_tranformation_config = DataTransformConfig()

    def get_data_transformer_obj(self):
        ''''
        this fucntion is responsible for data transformation based on the diffrent types of data
        '''
        try:
            numerical_columns = ['writing_score','reading_score']
            categorical_columns = ['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            num_pipeline =Pipeline(steps=[("imputer",SimpleImputer(strategy="median")), ##2steps, imputer(handeling missing values) simple imputer startegy will be median kyoki outlyers hai eda mai dikha tha
                                        ("scalar",StandardScaler(with_mean=False)) ] )                 #standard scalar
                                                                                        # aur yeah pipeline ko training data set pe run karegey fit_transform(training dataset)

            cat_pipeline=Pipeline(steps=[("imputer",SimpleImputer(strategy="most_frequent")),# most frequnt matlab mode
                                         ("one_hot_encode",OneHotEncoder()),
                                         ("standard_Scalar",StandardScaler(with_mean=False))   ])  
            
            ''' HERE WE ADDED WITH MEAN FLASE TO STANDNDARD SCALAR VIDEO MAI NAI HAI.....TRY RESOLVING IT'''
            
            logging.info("numerical  columns standard scaling completed")    
            logging.info("categorical columns encoding completed")

            # combine the numerical pipline with the categorical pipeline for that we will be using column transformer
            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns), # pehley name then pipeline wala variable then wo num columns jinmey apko column transformation chaiye 
                                                                    # do double tap on variable ki kaha sey aa rhe hai wo 

                    ("cat_pipline",cat_pipeline,categorical_columns)# same uper jaisa
                ]
            )

            return preprocessor  # so basically we have the whole preproccesd data here

        except Exception as e:
            raise CustomException(e,sys)
        
# now we will begin data transformation technique

    def initiate_data_transformation(self,train_path,test_path):                         # train & test hamko data ingestion sey aa rha hai
        try:
            train_df=pd.read_csv(train_path)                                            #reading data
            test_df=pd.read_csv(test_path)

            logging.info("read train & test data read")                                 #thoda logs

            logging.info("obtaining preprocessin gobject")

            preprocessing_obj = self.get_data_transformer_obj()                         #this function ko thoda rename kiya and has the 
                                                                                        #return value within which column transform,imputer
                                                                                        #then std scalar and onehot enoding
                                                                                        #the transformed columns

            target_column_name="math_score"                                             # target y
            numerical_columns = ['writing_score','reading_score']                       # kuch use mai to ani aya

            input_feaure_train_df=train_df.drop(columns=[target_column_name],axis=1)    #independant data, dropping y 
            target_feature_train_df=train_df[target_column_name]                        # yeah sirf y hai

            input_feaure_test_df=test_df.drop(columns=[target_column_name],axis=1)      #test data same uper ki tarha bus test purpose ke liye
            target_feature_test_df=test_df[target_column_name]

            logging.info("applying the preprocessing object on training dataframe and testing data frame")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feaure_train_df)   #yeah wala part thoda wapis samjhna padega
            input_feaure_test_arr = preprocessing_obj.transform(input_feaure_test_df)         #pre


            train_arr = np.c_[input_feature_train_arr,np.array(target_feature_train_df)]    #np.c_ sab aray ko eak 1d array mai reaarange kar deta 
            test_arr = np.c_[input_feaure_test_arr,np.array(target_feature_test_df)]        #and sabhi eak key niche eak lag jate hai 

            logging.info(f"saved preprocessing object")

            save_object(
                file_path=self.data_tranformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )   

            return(
                train_arr, test_arr, self.data_tranformation_config.preprocessor_obj_file_path  #pickel file end mai
            )
        except Exception as e:
            raise CustomException (e,sys)
    
    