import pandas as pd

df = pd.read_csv("data/fake_reviews_dataset.csv")

# CG = computer-generated/fake (1), OR = original/real (0)
df["label"] = (df["label"].str.strip().str.upper() == "CG").astype(int)
df["review"] = df["text_"].astype(str)

df = df[["review", "label", "category", "rating"]].dropna()
df.to_csv("data/reviews_clean.csv", index=False)

print(f"Ready: {len(df)} rows")
print(f"Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")
