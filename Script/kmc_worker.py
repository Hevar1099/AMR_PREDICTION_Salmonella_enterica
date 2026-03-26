import os
import subprocess
import mmh3


def process_single_genome(args):
    """
    Worker function running in a separate process.
    Self-contained: Has its own imports and variables.
    """
    path, row_idx, temp_dir, K, MEM_GB, N_FEAT = args
    genome_id = f"genome_{row_idx}"
    db_prefix = os.path.join(temp_dir, genome_id)
    kmc_cmd = ["kmc", f"-k{K}", f"-m{MEM_GB}", "-ci1", "-fm", path, db_prefix, temp_dir]
    try:
        subprocess.run(
            kmc_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError:
        return None
    # 2. RUN KMC_TOOLS
    dump_file = f"{db_prefix}.txt"
    dump_cmd = ["kmc_tools", "transform", db_prefix, "dump", dump_file]
    try:
        subprocess.run(dump_cmd, check=True, stdout=subprocess.DEVNULL)
    except subprocess.CalledProcessError:
        return None
    # 3. HASHING
    rows = []
    cols = []
    data = []
    try:
        with open(dump_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) != 2:
                    continue
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
        if os.path.exists(dump_file):
            os.remove(dump_file)
        for ext in [".kmc_pre", ".kmc_suf"]:
            if os.path.exists(db_prefix + ext):
                os.remove(db_prefix + ext)
    return rows, cols, data
