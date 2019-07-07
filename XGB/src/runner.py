import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import xgboost as xgb
import pickle
from sklearn.model_selection import learning_curve, GridSearchCV 
import matplotlib.pyplot as plt


class ForecastRunner(object):
    def __init__(self, filename, output_file, predicted_date):
        self.filename = filename
        self.output_file = output_file
        self.predicted_date = predicted_date


    def get_input(self):
        input_data = pd.read_csv(self.filename, index_col=0)
        return input_data

    def save_output(self, test, preds):
        preds = preds.reset_index(drop=True)
        df_test = test.reset_index()[['Date']]
        prediction = df_test.join(preds)
        prediction.to_csv(self.output_file)

    def prepare_data(self, df):
        df['Date'] = pd.to_datetime(df['Date'])
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Week'] = df['Date'].dt.week
        df['DOW'] = df['Date'].dt.weekday  
        start_date = pd.to_datetime(self.predicted_date).date()
        end_date = start_date + timedelta(days=6)
        df = df.set_index('Date')
        train = df.loc[df.index.date < start_date]
        test = df.loc[(df.index.date >= start_date) & (df.index.date <= end_date)]
        return train, test
    
    @staticmethod
    def grid_search(xtr, ytr):
        gbm = xgb.XGBRegressor()
        reg_cv = GridSearchCV(gbm, {"colsample_bytree":[0.9],"min_child_weight":[0.8,1.2]
                                ,'max_depth': [3,4,6], 'n_estimators': [500,1000], 'eval_metric':['rmse']}, verbose=1 )
        reg_cv.fit(xtr, ytr)
        return reg_cv

    @staticmethod
    def mean_absolute_percentage_error(y_true, y_pred): 
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    @staticmethod
    def plot_result(y_true, y_pred):
        plt.plot(y_true, label='Actual')
        plt.plot(y_pred, label='Predicted')
        plt.legend()
        plt.savefig('plot.png')
        
    def fit(self):
        """
        Gets data and preprocess by prepareData() function
       
        """
        today = datetime.now().date()
        data = self.get_input()
        df_train, df_test = self.prepare_data(data)
        xtr, ytr =  df_train.drop(['Value'], axis=1), df_train['Value'].values
        
        xgbtrain = xgb.DMatrix(xtr, ytr)
        reg_cv = ForecastRunner.grid_search(xtr, ytr)
        param = reg_cv.best_params_
        bst = xgb.train(dtrain=xgbtrain, params=param)
     
        # save model to file
        pickle.dump(bst, open("forecast.pickle.dat", "wb"))
        return df_test
    
    def predict(self, df_test):
        """
         Makes prediction for the next 7 days electricity consumption.
        """
        # load model from file
        loaded_model = pickle.load(open("forecast.pickle.dat", "rb"))
        # make predictions for test data
        xts, yts = df_test.drop(['Value'], axis=1), df_test['Value'].values
        p = loaded_model.predict(xgb.DMatrix(xts))
        prediction = pd.DataFrame({'Prediction': p})
        mape = ForecastRunner.mean_absolute_percentage_error(yts, p)
        print('MAPE: {}'.format(mape))
        ForecastRunner.plot_result(yts, p)
        self.save_output(df_test, prediction)

        
