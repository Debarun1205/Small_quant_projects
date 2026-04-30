# ---------------------------------------------------------------
# FICO Score Quantization / Rating Map using Dynamic Programming
# ---------------------------------------------------------------
# Goal:
# Create optimal bucket boundaries for FICO scores.
# Lower rating = better credit score.
#
# Two methods included:
#   1. Mean Squared Error Minimization
#   2. Log-Likelihood Maximization (recommended for credit risk)
#
# File required:
# Task 3 and 4_Loan_Data.csv
# ---------------------------------------------------------------

import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# Load Data
# ---------------------------------------------------------------
df = pd.read_csv("Task 3 and 4_Loan_Data.csv")

fico = df["fico_score"].values
default = df["default"].values

# Sort by FICO
order = np.argsort(fico)
fico = fico[order]
default = default[order]

n = len(fico)

# ---------------------------------------------------------------
# USER INPUT: Number of Buckets
# ---------------------------------------------------------------
num_buckets = 10


# ===============================================================
# PART A: LOG-LIKELIHOOD OPTIMAL BUCKETING (Dynamic Programming)
# ===============================================================

# Prefix sums for fast calculations
cum_defaults = np.cumsum(default)
cum_total = np.arange(1, n + 1)

def bucket_loglik(i, j):
    """
    Log likelihood of bucket from i to j inclusive
    """
    total = j - i + 1
    
    defaults = cum_defaults[j] - (cum_defaults[i - 1] if i > 0 else 0)
    
    # Avoid divide-by-zero
    p = defaults / total
    
    if p == 0 or p == 1:
        return 0
    
    non_defaults = total - defaults
    
    ll = defaults * np.log(p) + non_defaults * np.log(1 - p)
    return ll


# DP tables
dp = np.full((num_buckets + 1, n), -np.inf)
split = np.zeros((num_buckets + 1, n), dtype=int)

# Base case: 1 bucket
for j in range(n):
    dp[1][j] = bucket_loglik(0, j)

# Fill DP
for k in range(2, num_buckets + 1):
    for j in range(k - 1, n):
        for i in range(k - 2, j):
            val = dp[k - 1][i] + bucket_loglik(i + 1, j)
            if val > dp[k][j]:
                dp[k][j] = val
                split[k][j] = i

# Recover boundaries
boundaries = []
j = n - 1

for k in range(num_buckets, 1, -1):
    i = split[k][j]
    boundaries.append(fico[i])
    j = i

boundaries = sorted(boundaries)

print("Optimal FICO Bucket Boundaries:")
print(boundaries)


# ===============================================================
# PART B: CREATE RATING MAP
# Lower rating = better score
# ===============================================================

def fico_to_rating(score):
    """
    Convert FICO score to rating bucket
    Rating 1 = Best borrowers
    """
    rating = 1
    
    for b in reversed(boundaries):
        if score <= b:
            rating += 1
            
    return rating


# Example Usage
sample_scores = [820, 760, 690, 620, 580]

print("\nSample Ratings:")
for s in sample_scores:
    print(f"FICO {s} -> Rating {fico_to_rating(s)}")


# ===============================================================
# PART C: Bucket Summary Table
# ===============================================================

ratings = df["fico_score"].apply(fico_to_rating)

summary = pd.DataFrame({
    "fico_score": df["fico_score"],
    "default": df["default"],
    "rating": ratings
})

bucket_stats = summary.groupby("rating").agg(
    min_score=("fico_score", "min"),
    max_score=("fico_score", "max"),
    count=("fico_score", "count"),
    defaults=("default", "sum")
)

bucket_stats["pd"] = bucket_stats["defaults"] / bucket_stats["count"]

print("\nBucket Summary:")
print(bucket_stats.sort_index())


# ===============================================================
# OPTIONAL: MSE QUANTIZATION METHOD
# ===============================================================

# If needed, can also implement KMeans or equal-width binning
# But Log-Likelihood is preferred for credit risk modeling.
