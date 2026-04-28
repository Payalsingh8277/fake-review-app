import pandas as pd

# Load JSON correctly
df = pd.read_json("data/yelp_academic_dataset_review.json", lines=True, nrows=5000)

# Rename column
df = df.rename(columns={"text": "review"})

# Create label
df["label"] = df["stars"].apply(lambda x: 1 if x >= 4 else 0)

# Keep required columns
df = df[["review", "label"]].dropna()

# Save cleaned file
df.to_csv("data/reviews.csv", index=False)

print(f"Ready: {len(df)} rows")