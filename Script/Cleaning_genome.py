import pandas as pd
import re

df = pd.read_csv("Data/cleaned_metadata.csv")


def clean_genome_data(gene_string):
    if pd.isna(gene_string):
        return []
    genes = [g.split("=")[0].strip() for g in gene_string.split(",")]
    return genes


gene_matrix = (
    df["AMR genotypes"].apply(clean_genome_data).str.join("|").str.get_dummies()
)

final_dataset = pd.concat([df[["Assembly", "label", "Serovar"]], gene_matrix], axis=1)


def clean_serovar(name):
    if pd.isna(name):
        return "Unknown"
    name = str(name).strip()
    name = name.replace('"', "").replace("'", "")
    name = re.sub(r"^S\.\s+", "", name)
    name = re.sub(r"(?i)serovar\s+", "", name)
    if "var." in name:
        name = name.split("var.")[0].strip()
    if name.startswith("I ") or ":" in name:
        return "Monophasic Variant"
    return name.capitalize()


print("Cleaning Serovar names...")
final_dataset["Serovar"] = final_dataset["Serovar"].apply(clean_serovar)
final_dataset = final_dataset[final_dataset["Serovar"] != "Unknown"].copy()

print("Filtering lineage markers...")
gene_cols = gene_matrix.columns.tolist()
variances = final_dataset[gene_cols].var()
low_variance_genes = variances[variances < 0.01].index.tolist()

to_drop = []
for gene in gene_cols:
    if gene in low_variance_genes:
        to_drop.append(gene)
        continue

    serovar_means = final_dataset.groupby("Serovar")[gene].mean()
    is_lineage_marker = (serovar_means > 0.95).any()

    if is_lineage_marker:
        correlation = abs(final_dataset[gene].corr(final_dataset["label"]))
        if correlation < 0.1:
            to_drop.append(gene)

final_dataset = final_dataset.drop(columns=to_drop)

print(f"Final feature count: {len(final_dataset.columns) - 3}")
final_dataset.to_csv("Data/final_ml_dataset.csv", index=False)
