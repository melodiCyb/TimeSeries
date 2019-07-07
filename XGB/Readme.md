# Project Description

Irish electricity consumption forecasting for the next 7-days using XGBoost.

**Data**

Data contains date and daily consumption value. (Source: https://data.gov.ie/dataset/energy-consumption-gas-and-electricity-civic-offices-2009-2012/resource/6091c604-8c94-4b44-ac52-c1694e83d746)


**Requirements**
* xgboost



**Example run**

Usage: python main.py <source_file> <output_file> <predicted_date>

Example: python main.py ../data.csv ../output.csv 2013-01-07

