import pandas as pd

df = pd.read_csv("data/fake_reviews_dataset.csv")

# OR = fake (1), CG = genuine (0)
df["label"] = (df["label"].str.strip() == "OR").astype(int)
df["review"] = df["text_"].astype(str)

df = df[["review", "label", "category", "rating"]].dropna()
df.to_csv("data/reviews_clean.csv", index=False)

print(f"Ready: {len(df)} rows")
print(f"Fake: {df['label'].sum()} | Real: {(df['label']==0).sum()}")