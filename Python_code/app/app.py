

import plotly
import plotly.graph_objs as go

import pandas as pd
import os
import seaborn as sns
sns.set()

from config import DATA_PREPROCESSED_DIR

data = pd.read_csv("/home/michal/Desktop/EEG2/EEG_Classification_ANN_BEST/EEG_Classification_ANN/Python_code/dataset/preprocessed/subject2.csv")

data = [go.Scatter(
        x=list(range(0,len(data[column]))), # assign x as the dataframe column 'x'
        y=data[column],
        name=str(column)) for column in data.columns[1:-1]]


fig = plotly.offline.plot(data, filename='pandas-line-naming-traces.html',)





