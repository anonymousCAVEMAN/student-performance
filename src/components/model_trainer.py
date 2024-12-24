# here we will be specifically training our modle
# all the training code here all different kind of modules we will be using
# all the o/p  values will be assed over here
import os
import sys
from dataclasses import dataclass

from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRFRegressor
from sklearn.ensemble import (AdaBoostRegressor,GradientBoostingRegressor,RandomForestRegressor)
from catboost import CatBoostRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_models

from src.components.data_transfrormation import DataTransformation

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts',"model.pkl")

class ModelTrainer:
    def __init__(self) :
        self.model_trainer_config= ModelTrainerConfig()

    def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("split training and test input data")
            X_train,y_train,X_test,y_test = (train_array[:,:-1],# X_train mai train array ki all rows and last columnm sey all columns consider karo issey last column reh jayeggi
                                              train_array[:,-1],#all rows and only last column
                                              test_array[:,:-1],#all rowsn and columns bus begin with -1 jisse sey last ka column reh jayega
                                              test_array[:,-1])
            models ={
                "Decision Tree":DecisionTreeRegressor(),
                "Random forest":RandomForestRegressor(),                  
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression":LinearRegression(),
                "XGBRegressor":XGBRFRegressor(),
                "CatBoosting Regressor":CatBoostRegressor(),
                "ADAboost regressor":AdaBoostRegressor(),
                "K-Nearest Regressor":KNeighborsRegressor(),
            }

            model_report : dict = evaluate_models(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models)
            logging.info("began model evaluation")
            # best model score
            best_model_score = max(sorted(model_report.values()))  # model report basicaly evalution models ki retun value store kar rha hai
                                                                        # waha sey report return ho rha hai jo 
                                                                        # here we have 
                                                                    # max(sorted(list.....) # remove kiya hai
            # name of best model
            best_model_name = list(model_report.keys())[        # jasie wo utils wale for loop mai values call kiya ha index i ko dekey 
                                list(model_report.values()).      # wasie hi yaha keys call kiya hai and uska index best_model_score sey 
                                index(best_model_score)]          # liya hai list ke andar nested list create kiya hai
            
            best_model = models[best_model_name]

            if best_model_score<0.6 :
                raise CustomException ("no best model found")
            logging.info(f"best found model on both traing and testing data")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model   # this has the best model name saved in picklefile
                )
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test,predicted)
            return r2_square,best_model_name
    
        except Exception as e:
            raise CustomException (e,sys)


'''
 def initiate_model_trainer(self,train_array,test_array):
        try:
            logging.info("Split training and test input data")
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }
            params={
                "Decision Tree": {
                    'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                    # 'splitter':['best','random'],
                    # 'max_features':['sqrt','log2'],
                },
                "Random Forest":{
                    # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
                    # 'max_features':['sqrt','log2',None],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
                    'learning_rate':[.1,.01,.05,.001],
                    'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
                    # 'criterion':['squared_error', 'friedman_mse'],
                    # 'max_features':['auto','sqrt','log2'],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression":{},
                "XGBRegressor":{
                    'learning_rate':[.1,.01,.05,.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "CatBoosting Regressor":{
                    'depth': [6,8,10],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'iterations': [30, 50, 100]
                },
                "AdaBoost Regressor":{
                    'learning_rate':[.1,.01,0.5,.001],
                    # 'loss':['linear','square','exponential'],
                    'n_estimators': [8,16,32,64,128,256]
                }
                
            }

            model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
                                             models=models,param=params)
            
            ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))

            ## To get best model name from dict

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No best model found")
            logging.info(f"Best found model on both training and testing dataset")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted=best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            return r2_square
            



'''