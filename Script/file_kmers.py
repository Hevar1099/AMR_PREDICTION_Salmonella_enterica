import pandas as pd
import os
import glob

df = pd.read_csv("Data/final_ml_dataset.csv")
target_ids = df["Assembly"].unique()

search_dir = "Data/Genome/ncbi_dataset/data"
found_files = {}
for root, dirs, files in os.walk(search_dir):
    for file in files:
        if file.endswith(".fna") or file.endswith(".fasta"):
            for target in target_ids:
                if target in file:
                    full_path = os.path.join(root, file)
                    found_files[target] = full_path
                    break

path_df = pd.DataFrame(list(found_files.items()), columns=["Assembly", "Path"])
path_df.to_csv("Data/dna_file_paths.csv", index=False)
