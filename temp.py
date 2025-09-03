import pandas as pd
import matplotlib.pyplot as plt

# Load your CSV (replace 'your_file.csv' with the actual filename)
df = pd.read_csv("oban_2018_1_intensity90.csv")
df = df.clip(upper=200000)

# Show as heatmap
plt.imshow(df, cmap="hot", aspect="auto")
plt.colorbar(label="Value")
plt.title("Heatmap of CSV Data")
plt.show()
