import pandas as pd

df = pd.read_csv("data/annotated_responses.csv")

accuracy_score = df["Accuracy"].value_counts(normalize=True) * 100

print("\nAccuracy Distribution:")
print(accuracy_score)

error_cases = df[df["Accuracy"] == "Incorrect"]
print("\nIncorrect Cases:")
print(error_cases[["prompt", "response", "Comments"]])
