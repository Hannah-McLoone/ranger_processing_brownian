import numpy as np
import pandas as pd

park = 'oban'



def split(df, test_size=0.2):
    number_of_test_blocks = round(test_size * (len(df['block_id'].unique())))
    test_blocks = np.random.choice(df['block_id'].unique(), size=number_of_test_blocks, replace=False)
    mask = df['block_id'].isin(test_blocks)
    
    test = df[mask]
    train = df[~mask]
    return train, test


csv_files = []

for y in range(2016,2025):
    for m in range (1,12):
        csv_files.append(f'/maps/hm708/observations/{park}_{y}_{m}_observations.csv')



df = pd.concat([pd.read_csv(f) for f in csv_files], ignore_index=True)

train, test = split(df)
train.to_csv(f"/maps/hm708/final/{park}_train2.csv", index=False)
test.to_csv(f"/maps/hm708/final/{park}_test2.csv", index=False)
