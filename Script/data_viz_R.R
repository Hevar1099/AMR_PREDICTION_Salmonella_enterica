library(ComplexHeatmap)
library(circlize)

INPUT_DIR <- "Output/"
OUTPUT_DIR <- "Output/"
mat_path <- paste0(INPUT_DIR, "top_genes_matrix.csv")
meta_path <- paste0(INPUT_DIR, "metadata_for_heatmap.csv")

mat_df <- read.csv(mat_path, row.names = 1, check.names = FALSE)
mat <- as.matrix(mat_df)
meta <- read.csv(meta_path)

common_samples <- intersect(colnames(mat), meta$Assembly)
mat <- mat[, common_samples, drop = FALSE]
meta <- meta[match(common_samples, meta$Assembly), ]

print(paste("Plotting", nrow(mat), "genes across", ncol(mat), "samples..."))

# --- CLEAN UP THE LEGEND ---
serovar_counts <- table(meta$Serovar)
top_serovars <- names(sort(serovar_counts, decreasing = TRUE))[1:7]


meta$Serovar_Plot <- ifelse(meta$Serovar %in% top_serovars, meta$Serovar, "Other")

okabe_ito <- c("#56B4E9", "#E69F00", "#009E73", "#CC79A7", "#0072B2", "#D55E00", "#F0E442")

# Map colors to the Top 7 serovars
unique_plot_serovars <- unique(meta$Serovar_Plot)
palette_colors <- setNames(okabe_ito[1:length(top_serovars)], top_serovars)
palette_colors["Other"] <- "#999999" # Standard Okabe-Ito Grey for "Other"


pdf_path <- paste0(OUTPUT_DIR, "Figure4_OncoPrint.pdf")

tryCatch({
  pdf(pdf_path, width = 7.5, height = 5)

  ha_top <- HeatmapAnnotation(
    Phenotype = as.character(meta$label),
    Serovar = meta$Serovar_Plot,
    col = list(
      Phenotype = c("1" = "#D55E00", "0" = "#0072B2"), # Okabe-Ito Vermillion/Blue
      Serovar = palette_colors
    ),
    annotation_name_side = "left",
    annotation_name_gp = gpar(fontfamily = "Times", fontsize = 10),
    simple_anno_size = unit(0.4, "cm"),
    annotation_legend_param = list(
      Serovar = list(
        title = "Serovar",
        ncol = 1,
        title_gp = gpar(fontfamily = "Times", fontsize = 10, fontface = "bold"),
        labels_gp = gpar(fontfamily = "Times", fontsize = 9)
      ),
      Phenotype = list(
        title = "Phenotype",
        title_gp = gpar(fontfamily = "Times", fontsize = 10, fontface = "bold"),
        labels_gp = gpar(fontfamily = "Times", fontsize = 9)
      )
    )
  )

  col_fun <- c("0" = "whitesmoke", "1" = "#333333")

  ht <- Heatmap(mat,
    name = "Gene Presence",
    top_annotation = ha_top,
    col = col_fun,
    cluster_rows = TRUE,
    cluster_columns = TRUE,
    show_column_names = FALSE,
    show_row_names = TRUE,
    row_names_gp = gpar(fontfamily = "Times", fontsize = 10, fontface = "italic"),
    column_title = paste("Salmonella Assemblies (n=", ncol(mat), ")"),
    column_title_gp = gpar(fontfamily = "Times", fontsize = 12, fontface = "bold"),
    row_title = "Predictive AMR Features",
    row_title_gp = gpar(fontfamily = "Times", fontsize = 11, fontface = "bold"),
    heatmap_legend_param = list(
      title = "Presence",
      title_gp = gpar(fontfamily = "Times", fontsize = 10, fontface = "bold"),
      labels_gp = gpar(fontfamily = "Times", fontsize = 9)
    )
  )

  draw(ht, merge_legend = TRUE)
  print("Drawing complete.")
}, error = function(e) {
  print(paste("ERROR:", e$message))
}, finally = {
  dev.off()
  print(paste("Saved cleanly proportioned standard plot to:", pdf_path))
})
