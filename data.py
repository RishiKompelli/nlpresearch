import pandas as pd
from datasets import load_dataset
import spacy
import re
nlp = spacy.load("en_core_web_sm")


# dataset 1
ds = load_dataset("AnikaBasu/MentalHealthDataset")
df1 = pd.DataFrame(ds['train'])
df1 = df1.drop(['text', 'input'], axis=1)
df1.rename(columns={'instruction': 'input', 'output': 'output'}, inplace=True)
df1 = df1[['input', 'output']]
print("loaded dataset 1")
# dataset 2
df2 = pd.read_csv("hf://datasets/Riyazmk/mentalhealth/fullMentalHealth.csv")
df2.rename(columns={'Questions': 'input', 'Answers': 'output'}, inplace=True)
df2 = df2.loc[:, ~df2.columns.str.contains('^Unnamed')]
print("loaded dataset 2")
# dataset 3
df3 = pd.read_csv(r"C:\Users\rishi\OneDrive\Desktop\nlp research\dataset\data\train.csv")
df3.rename(columns={'Context': 'input', 'Response': 'output'}, inplace=True)
df3 = df3.loc[:, ~df3.columns.str.contains('^Unnamed')]
print("loaded dataset 3")
# dataset 4
df4 = pd.read_csv("hf://datasets/Kiran2004/MentalHealthConversations/Kiran-deppression.csv")
df4.rename(columns={'Questions': 'input', 'Answers': 'output'}, inplace=True)
df4 = df4.loc[:, ~df4.columns.str.contains('^Unnamed')]
print("loaded dataset 4")
# dataset 5
splits = {'train': 'train.csv', 'validation': 'validation.csv', 'test': 'test.csv'}
df5 = pd.read_csv("hf://datasets/Mr-Bhaskar/Synthetic_Therapy_Conversations/" + splits["train"])
df5.rename(columns={'human': 'input', 'ai': 'output'}, inplace=True)
df5 = df5.loc[:, ~df5.columns.str.contains('^Unnamed')]
print("loaded dataset 5")
# dataset 6
df6 = pd.read_parquet("hf://datasets/Aarya4536/therapy-bot-data-10k/data/train-00000-of-00001.parquet")
df6 = df6.drop(['response_k', 'text'], axis=1)
df6.rename(columns={'question': 'input', 'response_j': 'output'}, inplace=True)
print("loaded dataset 6")
# dataset 7
ds7 = load_dataset("adarshxs/Therapy-Alpaca")
df7 = pd.DataFrame(ds7['train'])
df7 = df7.drop('instruction', axis=1)
print("loaded dataset 7")
# dataset 8
df8 = pd.read_parquet("hf://datasets/mshojaei77/merged_mental_health_dataset/data/train-00000-of-00001.parquet")
df8.rename(columns={'Context': 'input', 'Response': 'output'}, inplace=True)
df8 = df8.loc[:, ~df3.columns.str.contains('^Unnamed')]
print("loaded dataset 8")

df = pd.concat([df1, df2, df3, df4, df5, df6, df7, df8], ignore_index=True)
print("Columns before dropping:", df.columns.tolist())
columns_to_drop = ['question', 'response_j']

# drop columns
for column in columns_to_drop:
    if column in df.columns:
        df = df.drop(column, axis=1)
df = df.drop_duplicates()
df = df.dropna(subset=['input', 'output'])

# remove names
df['input'] = df['input'].str.replace("Alex", '', regex=False)
df['output'] = df['output'].str.replace("Charlie", '', regex=False)

def fix_newlines(df, columns):
    for col in columns:
        df[col] = df[col].apply(lambda x: re.sub(r'\n', ' ', x) if isinstance(x, str) else x)
    return df

df = fix_newlines(df, ['input', 'output'])

df.to_csv("data.csv", index=False)