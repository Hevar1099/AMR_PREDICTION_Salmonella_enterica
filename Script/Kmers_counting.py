import pandas as pd
import numpy as np
import os
import time
import shutil
import sys
from scipy import sparse
import scipy.sparse
from concurrent.futures import ProcessPoolExecutor

# --- PART 1: AUTO-CREATE THE WORKER FILE ---
worker_code = """
import os
import subprocess
import mmh3

def process_single_genome(args):
    '''
    Worker function running in a separate process.
    Self-contained: Has its own imports and variables.
    '''
    path, row_idx, temp_dir, K, MEM_GB, N_FEAT = args
    
    genome_id = f"genome_{row_idx}"
    db_prefix = os.path.join(temp_dir, genome_id)
    
    # 1. RUN KMC (Counting)
    kmc_cmd = [
        "kmc",
        f"-k{K}",               
        f"-m{MEM_GB}",   
        "-ci1",                 
        "-fm",                  
        path,            
        db_prefix,          
        temp_dir                
    ]
    
    try:
        subprocess.run(kmc_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        return None

    # 2. RUN KMC_TOOLS (Dump to text)
    dump_file = f"{db_prefix}.txt"
    dump_cmd = ["kmc_tools", "transform", db_prefix, "dump", dump_file]
    
    try:
        subprocess.run(dump_cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    
    # 3. HASHING (Python Stream)
    rows = []
    cols = []
    data = []
    
    try:
        with open(dump_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2: continue
                
                kmer_seq, count = parts
                
                # Hashing Trick
                hash_val = mmh3.hash64(kmer_seq, seed=42, signed=False)[0]
                col_idx = hash_val % N_FEAT
                
                rows.append(row_idx)
                cols.append(col_idx)
                data.append(int(count))
    except Exception:
        return None
    finally:
        # Cleanup this worker's files
        if os.path.exists(dump_file): os.remove(dump_file)
        for ext in [".kmc_pre", ".kmc_suf"]:
            if os.path.exists(db_prefix + ext):
                os.remove(db_prefix + ext)

    return rows, cols, data
"""

# Write the worker file to disk
with open("kmc_worker.py", "w") as f:
    f.write(worker_code)

# IMPORT THE WORKER
try:
    from Script.kmc_worker import process_single_genome
except ImportError:
    # Fallback if path issues occur
    sys.path.append(os.getcwd())
    from Script.kmc_worker import process_single_genome


# --- PART 2: MAIN SCRIPT CONFIGURATION ---
N_FEATURES = 2**24  # 16.7 million features
K = 31  # K-mer length
KMC_MEMORY_GB = 1  # 1GB per job
JOBS = 8  # 8 Parallel Jobs
TEMP_DIR = "kmc_temp_data"


def check_dependencies():
    if shutil.which("kmc") is None:
        raise EnvironmentError(
            "KMC3 not found. Please install via 'conda install -c bioconda kmc'"
        )


def process_genomes_pipeline(paths):
    if not os.path.exists(TEMP_DIR):
        os.makedirs(TEMP_DIR)

    all_rows = []
    all_cols = []
    all_data = []

    print(f"🚀 Processing {len(paths)} genomes in PARALLEL (Jobs={JOBS})...")
    start_time = time.time()

    # Pack arguments for workers (must pass constants explicitly)
    tasks = [
        (path, i, TEMP_DIR, K, KMC_MEMORY_GB, N_FEATURES)
        for i, path in enumerate(paths)
    ]

    with ProcessPoolExecutor(max_workers=JOBS) as executor:
        # Submit all tasks
        futures = executor.map(process_single_genome, tasks)

        # Process results
        for i, result in enumerate(futures):
            if result:
                r, c, d = result
                all_rows.extend(r)
                all_cols.extend(c)
                all_data.extend(d)

            # Progress update
            if (i + 1) % 50 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                print(
                    f"   Processed {i + 1}/{len(paths)} ({rate:.2f} genomes/sec)...",
                    end="\r",
                )

    print("\n🔨 Constructing final sparse matrix...")

    coo = sparse.coo_matrix(
        (all_data, (all_rows, all_cols)),
        shape=(len(paths), N_FEATURES),
        dtype=np.uint32,
    )

    csr = coo.tocsr()

    if os.path.exists(TEMP_DIR):
        shutil.rmtree(TEMP_DIR)

    return csr


if __name__ == "__main__":
    check_dependencies()

    try:
        path_df = pd.read_csv("Data/dna_file_paths.csv")
        main_df = pd.read_csv("Data/final_ml_dataset.csv")
        merged_df = pd.merge(main_df[["Assembly", "label"]], path_df, on="Assembly")
        paths = merged_df["Path"].tolist()

        # Run Pipeline
        X_kmer = process_genomes_pipeline(paths)

        print(f"   Complete! Matrix Shape: {X_kmer.shape}")

        print("Saving results...")
        scipy.sparse.save_npz("Data/X_kmer_matrix.npz", X_kmer)
        merged_df.to_csv("Data/y_kmer_labels.csv", index=False)
        print("Done.")

    except Exception as e:
        print(f"\n❌ Execution Failed: {e}")
