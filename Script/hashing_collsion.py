import numpy as np


def hashing_collision_audit(n_unique_kmers, vector_size=2**24):
    """
    Calculates the theoretical collision rate based on the
    expected number of unique features vs bucket size.
    """
    # Theoretical collision rate formula: 1 - ( (M-1)/M )^(N(N-1)/2)

    collision_prob = 1 - np.exp(-(n_unique_kmers**2) / (2 * vector_size))

    print(f"--- Hashing Collision Audit ---")
    print(f"Feature Space (M): {vector_size:,}")
    print(f"Estimated Unique k-mers (N): {n_unique_kmers:,}")
    print(f"Theoretical Collision Probability: {collision_prob:.6%}")

    if collision_prob < 0.05:
        print(
            "Verdict: Collision rate is negligible. Model failure is likely biological/structural."
        )
    else:
        print("Verdict: High collision risk. Increase vector size.")


hashing_collision_audit(n_unique_kmers=10_000_000)
