import numpy as np
import pandas as pd





def split(df, test_size=0.2):
    number_of_test_blocks = round(test_size * (df['block_id'].max() + 1))
    test_blocks = np.random.choice(df['block_id'].unique(), size=number_of_test_blocks, replace=False)
    mask = df['block_id'].isin(test_blocks)
    
    test = df[mask]
    train = df[~mask]
    return train, test


csv_files = []

for y in range(2016,2025):
    for m in range (1,12):
        csv_files.append(f'/maps/hm708/observations/oban_{y}_{m}_observations.csv')


df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)
train, test = split(df)
train.to_csv("/maps/hm708/final/oban_train.csv", index=False)
test.to_csv("/maps/hm708/final/oban_test.csv", index=False)
