### Imports and Libraries
# To read csv
import pandas as pd

# To apply TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

# To apply cosine_similarity
from sklearn.metrics.pairwise import cosine_similarity

# To apply NearestNeighbors
from sklearn.neighbors import NearestNeighbors

# To create chart
import matplotlib.pyplot as plt

# To implement GroupShuffleSplit
from sklearn.model_selection import GroupShuffleSplit

### Load and Standardise Dataset
# Load the original dataset
df = pd.read_csv("enron_spam.csv")

# Make a separate working copy
clean_df = df.copy()

# Rename the email column to text
clean_df = clean_df.rename(columns={"email": "text"})

# Keep only the columns required for model training
clean_df = clean_df[["text", "label"]]

# Check the result
print("Columns:", clean_df.columns.tolist())
print("\nFirst 10 records:")
print(clean_df.head(10))

### Check Missing Values
# Check for missing values
print("\nMissing values:")
print(clean_df.isnull().sum())


### Check Empty Emails
# Check for empty emails or emails containing only spaces
empty_emails = clean_df["text"].fillna("").str.strip().eq("")

print("\nNumber of empty emails:")
print(empty_emails.sum())


### Check Invalid Labels
# Check all unique labels
print("\nUnique labels:")
print(clean_df["label"].unique())

# Find rows with invalid labels
# ~means NOT, basically reverse the Boolean result here
invalid_labels = ~clean_df["label"].isin([0, 1])

print("\nNumber of invalid labels:")
print(invalid_labels.sum())


### Check Conflicting Labels
# Check whether identical emails have conflicting labels
label_counts = clean_df.groupby("text")["label"].nunique()

conflicting_emails = label_counts[label_counts > 1]

print("\nNumber of emails with conflicting labels:")
print(len(conflicting_emails))


### Detect Exact Duplicates
# Check for duplicate email texts
duplicate_emails = clean_df.duplicated(subset=["text"])

print("\nNumber of duplicate emails:")
print(duplicate_emails.sum())

# Display the first 10 duplicate records
print("\nFirst 10 duplicate records:")
print(clean_df[duplicate_emails].head(10))


### Remove Exact Duplicates
# Remove exact duplicate emails
clean_df = clean_df.drop_duplicates(subset=["text"])

# Reset row numbering
clean_df = clean_df.reset_index(drop=True)

# Check how many emails remain
print("\nRecords after removing duplicates:")
print(len(clean_df))


### Check Case and Whitespace Duplicates
# Normalise text to detect simple near-duplicates
normalized_text = (
    clean_df["text"]
    .str.lower()
    .str.split()
    .str.join(" ")
)

# Check how many emails become duplicates after normalisation
near_duplicates = normalized_text.duplicated()

print("\nNear-duplicates after normalisation:")
print(near_duplicates.sum())


### Check Punctuation-Based Duplicates
# Normalise text by removing punctuation
#r"[^\w\s]" is regular expression, pattern used for matching text
#meaning not(^) letters,digits and underscore(\w) and not whitespace(\s)
#regax interpret first string as a regex pattern(regular expression)
#replaced with ""
normalized_text2 = (
    clean_df["text"]
    .str.lower()
    .str.replace(r"[^\w\s]", "", regex=True)
    .str.split()
    .str.join(" ")
)

# Check duplicates after removing punctuation
punctuation_duplicates = normalized_text2.duplicated()

print("\nDuplicates after removing punctuation:")
print(punctuation_duplicates.sum())


### Inspect Matching Emails
# Show all emails belonging to duplicate groups
# To compare cases like re-sign vs resign
# Become same meaning after removing - which is wrong
# .duplicated(keep=False) marks every occurence of a duplicated value
# Including the first
# For inspection
all_duplicates = normalized_text2.duplicated(keep=False)

# Display the original text and labels
# .loc[] select rows and columns by labels or conditions
print("\nEmails with matching normalised text:")
print(clean_df.loc[all_duplicates, ["text", "label"]].to_string())



### Validate and Remove Normalised Duplicates
# Check for conflicting labels after normalisation
temp_df = clean_df.copy()
temp_df["normalized_text"] = normalized_text2

conflicts = temp_df.groupby("normalized_text")["label"].nunique()

conflict_count = (conflicts > 1).sum()

print("\nConflicting labels after normalisation:")
print(conflict_count)

# Only remove duplicates if no conflicts exist
# Keep rows that are not duplicated
# Reset the index to prevent old index from becoming additional column
# Stops the program if there are unresolved label conflicts
if conflict_count == 0:
    clean_df = clean_df[~punctuation_duplicates].copy()
    clean_df = clean_df.reset_index(drop=True)

    print("\nRecords after removing punctuation duplicates:")
    print(len(clean_df))
else:
    raise ValueError(
        "Conflicting labels found! Review them before removing duplicates."
    )

### Remove Escaped-Newline and Whitespace Duplicates Globally

# Some original emails contain literal \n / \r / \t sequences
# instead of actual line breaks, so ordinary .split() can miss duplicates.
# This key is used only to detect repeats; original email text is preserved.
def duplicate_key(value):
    value = value.lower()
    for escape_sequence in (r"\r\n", r"\n", r"\r", r"\t"):
        value = value.replace(escape_sequence, " ")
    return " ".join(value.split())

# Check that equivalent messages have consistent ham/spam labels.
comparison_keys = clean_df["text"].map(duplicate_key)
key_labels = clean_df.assign(_key=comparison_keys).groupby("_key")["label"].nunique()
if (key_labels > 1).any():
    raise ValueError("Conflicting labels in formatting-normalized duplicate groups")

# Remove all formatting repeats BEFORE computing TF-IDF or creating splits.
format_duplicates = comparison_keys.duplicated(keep="first")
print("\nAdditional escaped-newline formatting duplicates:", int(format_duplicates.sum()))
clean_df = clean_df.loc[~format_duplicates].copy().reset_index(drop=True)
print("Emails after global formatting deduplication:", len(clean_df))


### Convert Emails to TF-IDF Features
# Create the TF-IDF vectorizer
# Keep at most 50000 vocabulary features to limit the number of columns
# A term must appear in at least 2 different emails
# This is to exclude terms that occur in only one email
vectorizer = TfidfVectorizer(
    max_features=50000,
    min_df=2
)

# Convert email text into numerical vectors
# .fit() learns the vocabulary and calculates statistic for TF-IDF
# .transform() converts each email into a numerical vector
email_vectors = vectorizer.fit_transform(clean_df["text"])

# Display the resulting matrix dimensions
print("\nTF-IDF matrix shape:")
print(email_vectors.shape)


### Find Similar Pairs in First 500 Emails
# Select the first 500 email vectors
sample_vectors = email_vectors[:500]

# Calculate similarity between the 500 emails
similarity_matrix = cosine_similarity(sample_vectors)

# Set the similarity threshold
threshold = 0.90

# Find highly similar pairs
similar_pairs = []

for i in range(500):
    for j in range(i + 1, 500):
        if similarity_matrix[i, j] >= threshold:
            similar_pairs.append(
                (i, j, similarity_matrix[i, j])
            )

print("\nHighly similar pairs in first 500 emails:")
print("Total pairs:", len(similar_pairs))

# Display the first 10 pairs
for i, j, score in similar_pairs[:10]:
    print(f"Email {i} and Email {j}: {score:.3f}")

### Inspect a Similar Email Pair
# Inspect Email 24 and one highly similar email
# Display the first 500characters

print("\nEmail 24:")
print(clean_df.iloc[24]["text"][:500])

print("\nEmail 37:")
print(clean_df.iloc[37]["text"][:500])

print("\nTheir labels:")
print("Email 24:", clean_df.iloc[24]["label"])
print("Email 37:", clean_df.iloc[37]["label"])

### Group Similar Emails in Sample
# Give each email its own initial group
parent = list(range(500))

# Find the group leader of an email
# While loop follows parent connections until it reaches a group leader
def find(x):
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x

# Combine the groups of two similar emails
def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        parent[root_b] = root_a

# Connect all highly similar email pairs
for i, j, score in similar_pairs:
    union(i, j)

# Collect emails by group
groups = {}

for i in range(500):
    root = find(i)

    if root not in groups:
        groups[root] = []

    groups[root].append(i)

# Keep groups containing at least two emails
similar_groups = [
    group for group in groups.values()
    if len(group) > 1
]

print("\nNumber of similar email groups:")
print(len(similar_groups))

print("\nFirst 5 groups:")
for group in similar_groups[:5]:
    print(group)

### Check Labels Within Sample Groups
print("\nChecking labels in similar groups:")

mixed_label_groups = 0

# clean_df.iloc[group] selects the emails belonging to a
# particular similarity group
# ["label"].unique() finds their distinct labels
for group in similar_groups:
    labels = clean_df.iloc[group]["label"].unique()

    if len(labels) > 1:
        mixed_label_groups += 1

print("Groups containing both spam and ham:", mixed_label_groups)

### Search Similar Emails Across Full Dataset
# Create a nearest-neighbour search model
# Retrieve up to 11 neighbours per email
# compare email using cosine distance
# calculate distances directly
# allow scikit-learn to use all available CPU cores
nn = NearestNeighbors(
    n_neighbors=11,
    metric="cosine",
    algorithm="brute",
    n_jobs=-1
)

# Fit the model using our existing TF-IDF vectors
nn.fit(email_vectors)

# Search in small batches to reduce memory usage
#process emails in batches
batch_size = 100
similar_pairs_full = set()

for start in range(0, len(clean_df), batch_size):

    end = min(start + batch_size, len(clean_df))

    distances, indices = nn.kneighbors(
        email_vectors[start:end]
    )

    for row, neighbours in enumerate(indices):
        i = start + row

        for k, j in enumerate(neighbours):
            if i == j:
                continue

            similarity = 1 - distances[row, k]

            if similarity >= 0.90:
                pair = (min(i, j), max(i, j))
                similar_pairs_full.add(pair)

print("\nHighly similar pairs across full dataset:")
print(len(similar_pairs_full))

### Group Similar Emails Across Full Dataset
# Reset Union-Find for all emails
parent = list(range(len(clean_df)))

# Connect all detected similar email pairs
for i, j in similar_pairs_full:
    union(i, j)

# Collect emails into groups
groups_full = {}

for i in range(len(clean_df)):
    root = find(i)

    if root not in groups_full:
        groups_full[root] = []

    groups_full[root].append(i)

# Keep groups containing at least two emails
similar_groups_full = [
    group for group in groups_full.values()
    if len(group) > 1
]

print("\nSimilar groups across full dataset:")
print(len(similar_groups_full))

# Check the largest group
largest_group = max(
    similar_groups_full,
    key=len,
    default=[]
)

print("\nLargest group size:")
print(len(largest_group))

# Check groups containing both spam and ham
mixed_label_groups_full = 0

for group in similar_groups_full:
    labels = clean_df.iloc[group]["label"].unique()

    if len(labels) > 1:
        mixed_label_groups_full += 1

print("\nMixed-label groups across full dataset:")
print(mixed_label_groups_full)

### Assign Similarity Group IDs
# Find the group leader for every email
group_ids = [find(i) for i in range(len(clean_df))]

# Add group IDs to our DataFrame
clean_df["group_id"] = group_ids

# Check the new columns
print("\nDataset columns:")
print(clean_df.columns.tolist())

# Count unique groups, including single-email groups
print("\nTotal unique groups:")
print(clean_df["group_id"].nunique())

# Display a small sample
print("\nFirst 10 group assignments:")
print(clean_df[["label", "group_id"]].head(10))

### Analyse Class Distribution
# Count legitimate and spam emails
class_counts = clean_df["label"].value_counts()

print("\nClass distribution after cleaning:")
print(class_counts)

# Calculate percentages
class_percentages = clean_df["label"].value_counts(normalize=True) * 100

print("\nClass percentages:")
print(class_percentages.round(2))

### Visualise Class Distribution
# Define the labels and their counts
labels = ["Legitimate (0)", "Spam (1)"]
counts = [
    class_counts[0],
    class_counts[1]
]

# Create the bar chart
plt.figure(figsize=(7, 5))
plt.bar(labels, counts)

# Add chart title and axis labels
plt.title("Email Class Distribution After Cleaning")
plt.xlabel("Email Classification")
plt.ylabel("Number of Emails")

# Display the count above each bar
for i, count in enumerate(counts):
    plt.text(i, count + 150, str(count), ha="center")

# Save the chart
plt.savefig("class_distribution.png", dpi=300, bbox_inches="tight")

# Display the chart
plt.show()

### Analyse Email Length Distribution
# Count characters in each email
clean_df["email_length"] = clean_df["text"].str.len()

# Display overall email length statistics
print("\nEmail length statistics:")
print(clean_df["email_length"].describe())

# Compare email lengths for legitimate and spam emails
print("\nEmail length statistics by class:")
print(clean_df.groupby("label")["email_length"].describe())

### Visualise Email Length Distribution
# Separate email lengths by classification
ham_lengths = clean_df.loc[
    clean_df["label"] == 0, "email_length"
]

spam_lengths = clean_df.loc[
    clean_df["label"] == 1, "email_length"
]

# Create a box plot
plt.figure(figsize=(8, 5))
plt.boxplot(
    [ham_lengths, spam_lengths],
    tick_labels=["Legitimate (0)", "Spam (1)"],
    showfliers=False
)

plt.title("Email Length Distribution by Classification")
plt.xlabel("Email Classification")
plt.ylabel("Email Length (Characters)")

# Save and display
plt.savefig("email_length_distribution.png",
            dpi=300, bbox_inches="tight")
plt.show()



### Create Group-Aware Dataset Splits

def create_splits(data):

    # Create the first split: 70% training, 30% remaining
    splitter1 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.70,
        random_state=42
    )

    train_idx, remaining_idx = next(
        splitter1.split(data, groups=data["group_id"])
    )

    train = data.iloc[train_idx].copy()
    remaining = data.iloc[remaining_idx].copy()

    # Divide remaining data into validation and testing
    splitter2 = GroupShuffleSplit(
        n_splits=1,
        train_size=0.50,
        random_state=42
    )

    val_idx, test_idx = next(
        splitter2.split(remaining, groups=remaining["group_id"])
    )

    validation = remaining.iloc[val_idx].copy()
    test = remaining.iloc[test_idx].copy()

    return train, validation, test


# Create initial splits
train_df, validation_df, test_df = create_splits(clean_df)

# Display split sizes
print("\nDataset split sizes:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))



### Verify Class Distribution Across Splits

# Check label proportions in each dataset
for name, dataset in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df)
]:
    print(f"\n{name} class distribution:")

    print(dataset["label"].value_counts())

    print("\nPercentages:")
    print(
        (dataset["label"].value_counts(normalize=True) * 100)
        .round(2)
    )


### Check for Group Overlap Between Splits

# Get the unique group IDs from each dataset
train_groups = set(train_df["group_id"])
val_groups = set(validation_df["group_id"])
test_groups = set(test_df["group_id"])

# Check whether any group appears in multiple datasets
train_val_overlap = train_groups & val_groups
train_test_overlap = train_groups & test_groups
val_test_overlap = val_groups & test_groups

print("\nGroup overlap check:")
print("Training vs Validation:", len(train_val_overlap))
print("Training vs Testing:", len(train_test_overlap))
print("Validation vs Testing:", len(val_test_overlap))


### Resolve Cross-Split Similarity Until Clear

from sklearn.metrics.pairwise import linear_kernel

# Search all three pairings, not only training versus testing.
# The TF-IDF vectors above have L2-normalised rows, so their dot
# products equal cosine similarities (apart from rounding).
def find_cross_split_pairs(df_a, df_b, threshold=0.90):
    found = set()
    a_vectors = email_vectors[df_a.index]
    b_vectors = email_vectors[df_b.index]

    for start in range(0, a_vectors.shape[0], 100):
        end = min(start + 100, a_vectors.shape[0])
        similarities = linear_kernel(
            a_vectors[start:end], b_vectors, dense_output=False
        ).tocoo()

        for row, col, score in zip(
            similarities.row, similarities.col, similarities.data
        ):
            if score >= threshold:
                # Use original clean_df row IDs, not batch positions.
                i = int(df_a.index[start + row])
                j = int(df_b.index[col])
                found.add((min(i, j), max(i, j)))
    return found


# Repeat because regrouping and re-splitting can reveal new links.
# Do not export until the final splits pass the similarity checks.
max_rounds = 15
for attempt in range(1, max_rounds + 1):
    train_val = find_cross_split_pairs(train_df, validation_df)
    train_test = find_cross_split_pairs(train_df, test_df)
    val_test = find_cross_split_pairs(validation_df, test_df)

    print(f"\nSimilarity check round {attempt}:")
    print("Train vs Validation:", len(train_val))
    print("Train vs Test:", len(train_test))
    print("Validation vs Test:", len(val_test))

    missed_pairs = train_val | train_test | val_test
    if not missed_pairs:
        print("All cross-split similarity checks passed (threshold 0.90).")
        break

    before_groups = clean_df["group_id"].nunique()
    for i, j in missed_pairs:
        union(i, j)
    clean_df["group_id"] = [find(i) for i in range(len(clean_df))]
    after_groups = clean_df["group_id"].nunique()

    if after_groups == before_groups:
        raise RuntimeError("Similarity pairs found, but no groups merged. Check indices.")

    train_df, validation_df, test_df = create_splits(clean_df)
else:
    raise RuntimeError("Similarity checks still failed after 15 rounds; outputs not saved.")

# The three datasets must also have disjoint similarity groups.
train_groups = set(train_df["group_id"])
val_groups = set(validation_df["group_id"])
test_groups = set(test_df["group_id"])
assert not (train_groups & val_groups)
assert not (train_groups & test_groups)
assert not (val_groups & test_groups)

print("\nFinal dataset sizes:")
print("Training:", len(train_df))
print("Validation:", len(validation_df))
print("Testing:", len(test_df))


### Verify Updated Class Distribution

for name, dataset in [
    ("Training", train_df),
    ("Validation", validation_df),
    ("Testing", test_df)
]:
    print(f"\nUpdated {name} class distribution:")

    print(dataset["label"].value_counts())

    print("\nPercentages:")
    print(
        (dataset["label"].value_counts(normalize=True) * 100)
        .round(2)
    )


### Verify Global Formatting Duplicates Are Absent

# These should all be absent because formatting repeats were removed
# from the complete dataset before group-aware splitting.
all_keys = [duplicate_key(value) for value in clean_df["text"]]
assert len(all_keys) == len(set(all_keys)), "Formatting duplicates remain in clean_df"

### Save Final Dataset Splits

# Keep only the columns needed for model training
train_final = train_df[["text", "label"]].copy()
validation_final = validation_df[["text", "label"]].copy()
test_final = test_df[["text", "label"]].copy()

# Save the three datasets
train_final.to_csv("train.csv", index=False)
validation_final.to_csv("validation.csv", index=False)
test_final.to_csv("test.csv", index=False)

print("\nFinal datasets saved successfully!")
print("train.csv:", len(train_final), "emails")
print("validation.csv:", len(validation_final), "emails")
print("test.csv:", len(test_final), "emails")


### Verify Saved CSV Files

# Read the saved datasets
saved_train = pd.read_csv("train.csv")
saved_validation = pd.read_csv("validation.csv")
saved_test = pd.read_csv("test.csv")

# Check the number of rows and columns
print("\nSaved dataset shapes:")
print("Training:", saved_train.shape)
print("Validation:", saved_validation.shape)
print("Testing:", saved_test.shape)

# Check the column names
print("\nTraining columns:", saved_train.columns.tolist())
print("Validation columns:", saved_validation.columns.tolist())
print("Testing columns:", saved_test.columns.tolist())

# Check whether any values are missing
print("\nMissing values:")
print("Training:", saved_train.isnull().sum().sum())
print("Validation:", saved_validation.isnull().sum().sum())
print("Testing:", saved_test.isnull().sum().sum())


### Verify Saved Dataset Contents

# Check that saved data matches the original DataFrames
assert saved_train.equals(
    train_final.reset_index(drop=True)
)

assert saved_validation.equals(
    validation_final.reset_index(drop=True)
)

assert saved_test.equals(
    test_final.reset_index(drop=True)
)

# Ensure all cleaned emails are included
assert (
    len(saved_train)
    + len(saved_validation)
    + len(saved_test)
    == len(clean_df)
)

# Ensure only valid labels are present
for dataset in [saved_train, saved_validation, saved_test]:
    assert dataset["label"].isin([0, 1]).all()

# Verify that no formatting-based duplicates exist within or between CSVs.
all_saved_keys = []
for dataset in [saved_train, saved_validation, saved_test]:
    all_saved_keys.extend(dataset["text"].map(duplicate_key).tolist())
assert len(all_saved_keys) == len(set(all_saved_keys)), (
    "Formatting duplicates remain in saved CSVs"
)

print("\nFinal dataset verification passed!")
