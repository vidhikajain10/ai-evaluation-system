import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

st.title("AI Response Evaluation & Annotation System")

# Load Data
df = pd.read_csv("data/sample_responses.csv")

if "annotations" not in st.session_state:
    st.session_state.annotations = []

for index, row in df.iterrows():
    st.subheader(f"Prompt {row['id']}")
    st.write("**Prompt:**", row["prompt"])
    st.write("**Response:**", row["response"])

    accuracy = st.selectbox(
        f"Accuracy for ID {row['id']}",
        ["Correct", "Partial", "Incorrect"],
        key=f"acc_{index}"
    )

    relevance = st.selectbox(
        f"Relevance for ID {row['id']}",
        ["Relevant", "Irrelevant"],
        key=f"rel_{index}"
    )

    completeness = st.selectbox(
        f"Completeness for ID {row['id']}",
        ["Complete", "Incomplete"],
        key=f"comp_{index}"
    )

    st.session_state.annotations.append({
        "id": row["id"],
        "Accuracy": accuracy,
        "Relevance": relevance,
        "Completeness": completeness
    })

if st.button("Save Annotations"):
    annotated_df = pd.DataFrame(st.session_state.annotations)
    annotated_df.to_csv("data/annotated_responses.csv", index=False)
    st.success("Annotations saved!")

# Automated Quality Scoring
if st.button("Generate Quality Score"):
    annotated_df = pd.DataFrame(st.session_state.annotations)

    score_map = {"Correct": 2, "Partial": 1, "Incorrect": 0}
    annotated_df["Score"] = annotated_df["Accuracy"].map(score_map)

    avg_score = annotated_df["Score"].mean()
    st.write("### Average Quality Score:", round(avg_score, 2))

# Confusion Matrix (Demo Example)
if st.button("Show Confusion Matrix (Demo)"):
    # Simulated ground truth for demo
    y_true = ["Correct", "Correct", "Incorrect", "Correct"]
    y_pred = [a["Accuracy"] for a in st.session_state.annotations[:4]]

    cm = confusion_matrix(y_true, y_pred, labels=["Correct", "Partial", "Incorrect"])

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=["Correct", "Partial", "Incorrect"],
                yticklabels=["Correct", "Partial", "Incorrect"])
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    st.pyplot(fig)
