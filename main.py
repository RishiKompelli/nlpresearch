import pandas as pd
from datasets import load_dataset

#dataset 1
ds = load_dataset("AnikaBasu/MentalHealthDataset")
df1 = pd.DataFrame(ds['train'])
df1 = df1.drop('text', axis=1)
df1 = df1.drop('input', axis = 1)
df1.rename(columns = {'instruction':'Input'}, inplace = True)
df1.rename(columns = {'output':'Output'}, inplace = True)
new_order = ['Input', 'Output']
df1 = df1[new_order]

#dataset 2
df2 = pd.read_csv("hf://datasets/Riyazmk/mentalhealth/fullMentalHealth.csv")
df2.rename(columns = {'Questions':'Input'}, inplace = True)
df2.rename(columns = {'Answers':'Output'}, inplace = True)

#dataset 3
df3 = pd.read_csv(r"C:\Users\rishi\OneDrive\Desktop\nlp research\nlpresearch\data\train.csv")
df3.rename(columns = {'Context':'Input'}, inplace = True)
df3.rename(columns = {'Response':'Output'}, inplace = True)


df = pd.concat([df1, df2], ignore_index=True)
df = pd.concat([df, df3], ignore_index=True)
df = df.drop_duplicates()
print(df)