import pandas as pd

df1 = pd.read_csv("data/annotated_responses.csv")
df2 = df1.copy()

# Simulate second annotator with small variation
df2.loc[0, "Accuracy"] = "Partial"

agreement = (df1["Accuracy"] == df2["Accuracy"]).mean() * 100

print("Inter-Annotator Agreement:", round(agreement, 2), "%")
