import pandas as pd
from datasets import load_dataset

# dataset 1
ds = load_dataset("AnikaBasu/MentalHealthDataset")
df1 = pd.DataFrame(ds['train'])
df1 = df1.drop(['text', 'input'], axis=1)
df1.rename(columns={'instruction': 'Input', 'output': 'Output'}, inplace=True)
df1 = df1[['Input', 'Output']]

# dataset 2
df2 = pd.read_csv("hf://datasets/Riyazmk/mentalhealth/fullMentalHealth.csv")
df2.rename(columns={'Questions': 'Input', 'Answers': 'Output'}, inplace=True)
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]

# dataset 3
df3 = pd.read_csv(r"C:\Users\rishi\OneDrive\Desktop\nlp research\nlpresearch\data\train.csv")
df3.rename(columns={'Context': 'Input', 'Response': 'Output'}, inplace=True)
df3 = df3.loc[:, ~df3.columns.str.contains('^Unnamed')]

# dataset 4
df4 = pd.read_csv("hf://datasets/Kiran2004/MentalHealthConversations/Kiran-deppression.csv")
df4.rename(columns={'Questions': 'Input', 'Answers': 'Output'}, inplace=True)
df4 = df4.loc[:, ~df4.columns.str.contains('^Unnamed')]

# dataset 5
splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
df5 = pd.read_csv("hf://datasets/Mr-Bhaskar/Synthetic_Therapy_Conversations/" + splits["train"])
df5.rename(columns={'human': 'Input', 'ai': 'Output'}, inplace=True)
df5 = df5.loc[:, ~df5.columns.str.contains('^Unnamed')]

# dataset 6
df6 = pd.read_parquet("hf://datasets/Aarya4536/therapy-bot-data-10k/data/train-00000-of-00001.parquet")
df6 = df6.drop(['response_k', 'text'], axis=1)
df5.rename(columns={'question': 'Input', 'response_j': 'Output'}, inplace=True)

# dataset 7
ds = load_dataset("adarshxs/Therapy-Alpaca")
df7 = pd.DataFrame(ds['train'])
df7 = df7.drop('instruction', axis=1)
df7.rename(columns={'input': 'Input', 'output': 'Output'}, inplace=True)

df = pd.concat([df1, df2, df3, df4, df5, df6, df7], ignore_index=True)
df = df.drop(['question', 'response_j'], axis=1)
df = df.drop_duplicates()
df = df.dropna(subset = ['Input', 'Output'])
print(df)
