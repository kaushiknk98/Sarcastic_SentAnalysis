import csv


import pandas as pd
#df = pd.read_csv('train-balanced-sarcasm.csv', usecols=['label','parent_comment']).T.to_dict()
df = pd.read_csv('train-balanced-sarcasm.csv', usecols=['label','comment'], header=0)
val=54620
df1=df[:][val:val+150]
df=df[:][:val]
train_data=df.set_index('comment').to_dict()
train_data=train_data['label']
test_data=df1.set_index('comment').to_dict()
test_data=test_data['label']