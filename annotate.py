import pandas as pd

df = pd.read_csv("data/sample_responses.csv")

df["Accuracy"] = ""
df["Relevance"] = ""
df["Completeness"] = ""
df["Comments"] = ""

for i, row in df.iterrows():
    print("\nPrompt:", row["prompt"])
    print("Response:", row["response"])
    
    df.at[i, "Accuracy"] = input("Accuracy (Correct/Partial/Incorrect): ")
    df.at[i, "Relevance"] = input("Relevance (Relevant/Irrelevant): ")
    df.at[i, "Completeness"] = input("Completeness (Complete/Incomplete): ")
    df.at[i, "Comments"] = input("Comments: ")

df.to_csv("data/annotated_responses.csv", index=False)
print("Annotation saved.")
