import sys
import argparse
import pandas as pd
import numpy as np
from multiprocess import Pool
import pickle
import kde # make sure kde.py is present in directory

def make_argument_parser():
    """
    Creates an ArgumentParser to read the options for this script from
    sys.argv
    """
    parser = argparse.ArgumentParser(
        description="Create metadata",)
    parser.add_argument('--metadata_pkl_path', '-i', type=str, required=True,
                        help='Path to metada_bc.pkl file.')
    
    parser.add_argument('--cores', '-c', type=int, required=True,
                        help='Path to metada_bc.pkl file.')
    
    return parser

def main():
    """
    The main executable function
    """
    parser = make_argument_parser()
    
    args = parser.parse_args()
    metadata_pkl_path = args.metadata_pkl_path
    cores = args.cores
    out_path = metadata_pkl_path.replace(".pkl", "_kde.tsv")
    
    x_var = "Unique_nr_frag_in_regions"
    y_var_list = ["TSS_enrichment", "FRIP", "Dupl_rate"]
    
    with open(metadata_pkl_path, 'rb') as f:
        metadata_bc_df = pickle.load(f)
        
    x_log = np.log(metadata_bc_df[x_var] + 1)
    
    kde_df = pd.DataFrame(index=metadata_bc_df.index, columns=y_var_list)
    for y_var in y_var_list:
        y = metadata_bc_df[y_var]
        # Split input array for KDE [log(x), y] array in
        # equaly spaced parts (start_offset + n * nbr_cores).
        kde_parts = [np.vstack([x_log[i::cores], y[i::cores]]) for i in range(cores)]
        barcodes = [metadata_bc_df.index[i::cores] for i in range(cores)]
        barcodes = [bc for sublist in barcodes for bc in sublist]
        # print(barcodes)
        
        # Calculate KDE in parallel.
        with Pool(processes=cores) as pool:
            results = pool.map(kde.calc_kde, kde_parts)

        z = np.concatenate(results)

        # now order x and y in the same way that z was ordered, otherwise random z value is assigned to barcode:
        # x_ordered = np.concatenate([x_log[i::cores] for i in range(cores)])
        # y_ordered = np.concatenate([y[i::cores] for i in range(cores)])
        
        # order based on z value so that highest value is plotted on top, and not hidden by lower values
        # idx = z.argsort()
        # metadata_bc_sub = pd.DataFrame(index=metadata_bc_df.index[idx])
        # metadata_bc_sub["kde"] = z[idx]
        # metadata_bc_df[f"kde__log_{x_var}__{y_var}"] = metadata_bc_sub["kde"]
        
        metadata_bc_sub = pd.DataFrame(index=barcodes)
        metadata_bc_sub["kde"] = z
        print(metadata_bc_sub)
        
        kde_df[f"kde__log_{x_var}__{y_var}"] = metadata_bc_sub["kde"]
        print(f"{x_var}, {y_var} done")        
    kde_df.to_csv(out_path, sep='\t', index=True, header=True)
    
    with open(metadata_pkl_path, "wb") as f:
        pickle.dump(metadata_bc_df, f, protocol=4)
if __name__ == "__main__":
    main()