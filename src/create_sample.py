import pandas as pd

# Path
input_path = "../data/consumer_complaints.csv"    # full dataset
output_path = "../data/consumer_complaints_sample.csv"  # sampled dataset

# Number of rows you want in the sample
sample_size = 100_000

print("Reading dataset in chunks...")

# Use iterator to avoid memory issues
chunks = pd.read_csv(input_path, chunksize=100_000, low_memory=False)

sampled_df = pd.DataFrame()

for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}...")
    # Take random 2,000 rows from each chunk to reach ~100,000 total
    sampled_df = pd.concat([sampled_df, chunk.sample(n=2000, random_state=42)])

    if len(sampled_df) >= sample_size:
        break

# Save the final sample
sampled_df = sampled_df.sample(n=sample_size, random_state=42)  # ensure exact 100k
sampled_df.to_csv(output_path, index=False)

print(f"✅ Sampled file saved to {output_path}")
print("Shape:", sampled_df.shape)

# # Task 5 - Text Classification
# **Candidate Name:** Bathula Sai Teja  
# **Date/Time:** 2025-09-27
