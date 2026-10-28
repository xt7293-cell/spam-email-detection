# To read csv
import pandas as pd

# Read the dataset
df = pd.read_csv("enron_spam.csv")

# Check dataset structure
#df returns the dimensions of the DataFrame
print("Dataset shape:", df.shape)

# Converts column names to python list
print("\nColumns:", df.columns.tolist())

# Show the first 10 records
print("\nFirst 10 records:")
print(df.head(10).to_string())

# Check class distribution
print("\nLabel distribution:")
print(df["label"].value_counts())
