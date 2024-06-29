import pandas as pd
from datasets import load_dataset

ds = load_dataset("AnikaBasu/MentalHealthDataset")
df1 = pd.DataFrame(ds)
df2 = pd.read_csv("hf://datasets/Riyazmk/mentalhealth/fullMentalHealth.csv")
df3 = pd.read_csv(r"C:\Users\rishi\OneDrive\Desktop\nlp research\nlpresearch\data\train.csv")

pd.merge(df1, df2, on='text', how='inner')