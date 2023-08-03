# PUMATAC Tutorial
Tutorial for [PUMATAC (Pipeline for Universal Mapping of ATAC-seq)](https://github.com/aertslab/PUMATAC).  
Please follow all notebooks in order. Notebook 0 contains scripts that will download or generate all the resources you need in addition to Jupyter.

In notebook `5_qc_diagnosis.ipynb`, you can find a detailed and ready-to-run notebook that will reproduce the main quality plots in our manuscript, but for your data. The tool works out of the box with 10x Genomics Cell Ranger output. The plots include cell filtering based on # fragments and TSS, violin plots for #fragments, TSS enrichment and FRIP, the stacked bar plot for the fraction of reads lost at each stage, and saturation plots. The resulting graphs will look like so:

![image](https://github.com/aertslab/PUMATAC_tutorial/assets/55103921/deb9e44d-cde5-47ef-af40-84fdb1cb6187)

If you use this tool, please cite [our 2023 Nature Biotechnology article]([https://www.nature.com/articles/s41587-023-01881-x).
De Rop, F.V., Hulselmans, G., Flerin, C. et al. Systematic benchmarking of single-cell ATAC-sequencing protocols. Nat Biotechnol (2023). https://doi.org/10.1038/s41587-023-01881-x
