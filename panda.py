import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("profile_log.csv")
print(df.columns)
df["mem_after_mb"].plot()
plt.show()
