import pandas as pd

def getKeys():
  df = pd.read_csv("files/api_keys.csv")

  keys = df['API_KEYS'].tolist()

  return keys