import pandas as pd

file_path = r"data\consumer_complaints.csv"  # correct relative path
chunksize = 100_000

for i, chunk in enumerate(pd.read_csv(file_path, chunksize=chunksize)):
    out_file = fr"data\complaints_part_{i}.csv"
    chunk.to_csv(out_file, index=False)
    print(f"Saved: {out_file}")


# # Task 5 - Text Classification
# **Candidate Name:** Bathula Sai Teja  
# **Date/Time:** 2025-09-27
