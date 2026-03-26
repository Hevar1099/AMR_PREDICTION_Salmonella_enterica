import pandas as pd

meta_data = pd.read_csv("Data/metadata.tsv", sep="\t")

all_phenotypes = meta_data["AST phenotypes"].str.cat(sep=",")
all_drugs = pd.Series(drug.split("=")[0] for drug in all_phenotypes.split(","))

drug_counts = all_drugs.value_counts().sort_values(ascending=False)

print("Antibiotic frequency (most repeated first):")
print(drug_counts)


def get_cipro_status(ast_string):
    """Extracts the ciprofloxacin status from an AST phenotype string."""
    if pd.isna(ast_string):
        return None
    parts = ast_string.split(",")
    for p in parts:
        if "ciprofloxacin" in p:
            return p.split("=")[1]
    return None


meta_data["cipro_status"] = meta_data["AST phenotypes"].apply(get_cipro_status)
cipro_counts = meta_data["cipro_status"].value_counts()
print("\nCiprofloxacin status counts:")
print(cipro_counts)

cleaned_meta_data = meta_data[meta_data["cipro_status"].isin(["R", "S"])].copy()
cleaned_meta_data["label"] = cleaned_meta_data["cipro_status"].map({"R": 1, "S": 0})

cleaned_meta_data = cleaned_meta_data.dropna(subset=["Assembly"])
cleaned_meta_data.to_csv("Data/cleaned_metadata.csv", index=False)

cleaned_meta_data["Assembly"].to_csv(
    "Data/accessions_clean.txt", index=False, header=False
)


raw_df = cleaned_meta_data
genes_raw = (
    raw_df["AMR genotypes"]
    .str.get_dummies(sep=",")
    .rename(columns=lambda x: x.split("=")[0].strip())
)
genes_clean = genes_raw.groupby(level=0, axis=1).max()

print(f"Columns before: {genes_raw.shape[1]}")
print(f"Columns after: {genes_clean.shape[1]}")

target_proxy = "aph(3'')-Ib"
actual_mechanisms = ["qnrS1", "qnrB1", "qnrA1", "aac(6')-Ib-cr", "oqxAB"]

for mech in actual_mechanisms:
    if mech in genes_clean.columns:
        co_occurrence = (
            (genes_clean[target_proxy] == 1) & (genes_clean[mech] == 1)
        ).sum()
        total_proxy = genes_clean[target_proxy].sum()

        if total_proxy > 0:
            percentage = (co_occurrence / total_proxy) * 100
            print(
                f"{target_proxy} co-carriage with {mech}: {co_occurrence}/{total_proxy} ({percentage:.2f}%)"
            )
