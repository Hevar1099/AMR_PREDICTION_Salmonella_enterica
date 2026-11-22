import numpy as np


def hashing_collision_audit(n_unique_kmers, vector_size=2**24):
    """
    Calculates collision probability, expected bin occupancy,
    expected colliding k-mers, and weight-vector memory estimates.
    """
    lam = n_unique_kmers / vector_size
    collision_prob = 1 - np.exp(-(n_unique_kmers * (n_unique_kmers - 1)) / (2 * vector_size))

    p_empty = np.exp(-lam)
    p_singleton = lam * np.exp(-lam)
    p_colliding_bin = 1 - p_empty - p_singleton

    expected_occupied = vector_size * (1 - p_empty)
    expected_singletons = vector_size * p_singleton
    expected_colliding_bins = vector_size * p_colliding_bin
    frac_kmers_colliding = 1 - np.exp(-lam)
    expected_colliding_kmers = n_unique_kmers * frac_kmers_colliding

    print("--- Hashing Collision Audit ---")
    print(f"Feature Space (D): {vector_size:,}")
    print(f"Estimated Unique k-mers (n): {n_unique_kmers:,}")
    print(f"Load factor (lambda = n/D): {lam:.4f}")
    print(f"P(at least one collision): {collision_prob:.6%}")
    print(f"Expected occupied bins: {expected_occupied:,.0f} ({(1 - p_empty) * 100:.2f}% of D)")
    print(f"Expected singleton bins (1 k-mer): {expected_singletons:,.0f} ({p_singleton * 100:.2f}% of D)")
    print(
        f"Expected colliding bins (>=2 k-mers): {expected_colliding_bins:,.0f} "
        f"({p_colliding_bin * 100:.2f}% of D; {expected_colliding_bins / expected_occupied * 100:.2f}% of occupied bins)"
    )
    print(
        f"Expected k-mers involved in a collision: {expected_colliding_kmers:,.0f} "
        f"({frac_kmers_colliding * 100:.2f}% of n)"
    )

    print("\n--- Single float64 Weight-Vector Memory ---")
    for bits in (24, 28, 32):
        size_mb = (2**bits) * 8 / (1024**2)
        print(f"2^{bits} bins: {size_mb:,.1f} MB ({size_mb / 1024:.2f} GB)")


hashing_collision_audit(n_unique_kmers=10_000_000)
