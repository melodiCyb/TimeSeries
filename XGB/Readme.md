# Project Description

Irish electricity consumption forecasting for the next 7-days using XGBoost.

**Data**

Data contains electricty consumption measurements for 15-min intervals from March, 2011 to February 2013.

Source : [data.gov.ie](https://data.gov.ie/dataset/energy-consumption-gas-and-electricity-civic-offices-2009-2012/resource/6091c604-8c94-4b44-ac52-c1694e83d746)

To download: https://data.smartdublin.ie/dataset/10130831-d2a5-4f4f-b56b-03d8e2cc56c8/resource/6091c604-8c94-4b44-ac52-c1694e83d746/download/dccelectricitycivicsblocks34p20130221-1840.csv

**Requirements**
* xgboost



**Example run**

Usage: python main.py --url {url} --output_file {output_file} --predicted_date {predicted_date}
    --min_child_weight {min_child_weight} --colsample_bytree {colsample_bytree} --max_depth {max_depth}
    --n_estimators {n_estimators} --eval_metric {eval_metric}

