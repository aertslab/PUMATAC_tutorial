import os
import glob
import pprint as pp
import re
import gzip

import pycisTopic
from pycisTopic.qc import *
import pickle
import pybiomart as pbm
from scipy.optimize import curve_fit
import scipy

import numpy as np
import pandas as pd
import polars as pl

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from matplotlib.gridspec import GridSpec
import seaborn as sns
import palettable

# dictionary of instrument id regex: [platform(s)]
InstrumentIDs = {
    "HWI-M[0-9]{4}$": ["MiSeq"],
    "HWUSI": ["Genome Analyzer IIx"],
    "M[0-9]{5}$": ["MiSeq"],
    "HWI-C[0-9]{5}$": ["HiSeq 1500"],
    "C[0-9]{5}$": ["HiSeq 1500"],
    "HWI-D[0-9]{5}$": ["HiSeq 2500"],
    "D[0-9]{5}$": ["HiSeq 2500"],
    "J[0-9]{5}$": ["HiSeq 3000"],
    "K[0-9]{5}$": ["HiSeq 3000", "HiSeq 4000"],
    "E[0-9]{5}$": ["HiSeq X"],
    "NB[0-9]{6}$": ["NextSeq 500/550"],
    "NS[0-9]{6}$": ["NextSeq 500/550"],
    "MN[0-9]{5}$": ["MiniSeq"],
    "N[0-9]{5}$": ["NextSeq 500/550"],  # added since original was outdated
    "A[0-9]{5}$": ["NovaSeq 6000"],  # added since original was outdated
    "V[0-9]{5}$": ["NextSeq 2000"],  # added since original was outdated
    "VH[0-9]{5}$": ["NextSeq 2000"],  # added since original was outdated
}

# dictionary of flow cell id regex: ([platform(s)], flow cell version and yeild)
FCIDs = {
    "C[A-Z,0-9]{4}ANXX$": (
        ["HiSeq 1500", "HiSeq 2000", "HiSeq 2500"],
        "High Output (8-lane) v4 flow cell",
    ),
    "C[A-Z,0-9]{4}ACXX$": (
        ["HiSeq 1000", "HiSeq 1500", "HiSeq 2000", "HiSeq 2500"],
        "High Output (8-lane) v3 flow cell",
    ),
    "H[A-Z,0-9]{4}ADXX$": (
        ["HiSeq 1500", "HiSeq 2500"],
        "Rapid Run (2-lane) v1 flow cell",
    ),
    "H[A-Z,0-9]{4}BCXX$": (
        ["HiSeq 1500", "HiSeq 2500"],
        "Rapid Run (2-lane) v2 flow cell",
    ),
    "H[A-Z,0-9]{4}BCXY$": (
        ["HiSeq 1500", "HiSeq 2500"],
        "Rapid Run (2-lane) v2 flow cell",
    ),
    "H[A-Z,0-9]{4}BBXX$": (["HiSeq 4000"], "(8-lane) v1 flow cell"),
    "H[A-Z,0-9]{4}BBXY$": (["HiSeq 4000"], "(8-lane) v1 flow cell"),
    "H[A-Z,0-9]{4}CCXX$": (["HiSeq X"], "(8-lane) flow cell"),
    "H[A-Z,0-9]{4}CCXY$": (["HiSeq X"], "(8-lane) flow cell"),
    "H[A-Z,0-9]{4}ALXX$": (["HiSeq X"], "(8-lane) flow cell"),
    "H[A-Z,0-9]{4}BGXX$": (["NextSeq"], "High output flow cell"),
    "H[A-Z,0-9]{4}BGXY$": (["NextSeq"], "High output flow cell"),
    "H[A-Z,0-9]{4}BGX2$": (["NextSeq"], "High output flow cell"),
    "H[A-Z,0-9]{4}AFXX$": (["NextSeq"], "Mid output flow cell"),
    "A[A-Z,0-9]{4}$": (["MiSeq"], "MiSeq flow cell"),
    "B[A-Z,0-9]{4}$": (["MiSeq"], "MiSeq flow cell"),
    "D[A-Z,0-9]{4}$": (["MiSeq"], "MiSeq nano flow cell"),
    "G[A-Z,0-9]{4}$": (["MiSeq"], "MiSeq micro flow cell"),
    "H[A-Z,0-9]{4}DMXX$": (["NovaSeq"], "S2 flow cell"),
}


SUPERNOVA_PLATFORM_BLACKLIST = ["HiSeq 3000", "HiSeq 4000", "HiSeq 3000/4000"]

_upgrade_set1 = set(["HiSeq 2000", "HiSeq 2500"])
_upgrade_set2 = set(["HiSeq 1500", "HiSeq 2500"])
_upgrade_set3 = set(["HiSeq 3000", "HiSeq 4000"])
_upgrade_set4 = set(["HiSeq 1000", "HiSeq 1500"])
_upgrade_set5 = set(["HiSeq 1000", "HiSeq 2000"])

fail_msg = "Cannot determine sequencing platform"
success_msg_template = "(likelihood: {})"
null_template = "{}"

# do intersection of lists
def intersect(a, b):
    return list(set(a) & set(b))


def union(a, b):
    return list(set(a) | set(b))


# extract ids from reads
def parse_readhead(head):
    fields = head.strip("\n").split(":")

    # if ill-formatted/modified non-standard header, return cry-face
    if len(fields) < 3:
        return -1, -1
    iid = fields[0][1:]
    fcid = fields[2]
    return iid, fcid


# infer sequencer from ids from single fastq
def infer_sequencer(iid, fcid):
    seq_by_iid = []
    for key in InstrumentIDs:
        if re.search(key, iid):
            seq_by_iid += InstrumentIDs[key]

    seq_by_fcid = []
    for key in FCIDs:
        if re.search(key, fcid):
            seq_by_fcid += FCIDs[key][0]

    sequencers = []

    # if both empty
    if not seq_by_iid and not seq_by_fcid:
        return sequencers, "fail"

    # if one non-empty
    if not seq_by_iid:
        return seq_by_fcid, "likely"
    if not seq_by_fcid:
        return seq_by_iid, "likely"

    # if neither empty
    sequencers = intersect(seq_by_iid, seq_by_fcid)
    if sequencers:
        return sequencers, "high"
    # this should not happen, but if both ids indicate different sequencers..
    else:
        sequencers = union(seq_by_iid, seq_by_fcid)
        return sequencers, "uncertain"


# process the flag and detected sequencer(s) for single fastq
def infer_sequencer_with_message(iid, fcid):
    sequencers, flag = infer_sequencer(iid, fcid)
    if not sequencers:
        return [""], fail_msg

    if flag == "high":
        msg_template = null_template
    else:
        msg_template = success_msg_template

    if set(sequencers) <= _upgrade_set1:
        return ["HiSeq2000/2500"], msg_template.format(flag)
    if set(sequencers) <= _upgrade_set2:
        return ["HiSeq1500/2500"], msg_template.format(flag)
    if set(sequencers) <= _upgrade_set3:
        return ["HiSeq3000/4000"], msg_template.format(flag)
    return sequencers, msg_template.format(flag)


def test_sequencer_detection():
    Samples = [
        "@ST-E00314:132:HLCJTCCXX:6:2206:31213:47966 1:N:0",
        "@D00209:258:CACDKANXX:6:2216:1260:1978 1:N:0:CGCAGTT",
        "@D00209:258:CACDKANXX:6:2216:1586:1970 1:N:0:GAGCAAG",
        "@A00311:74:HMLK5DMXX:1:1101:2013:1000 3:N:0:ACTCAGAC",
    ]

    seqrs = set()
    for head in Samples:
        iid, fcid = parse_readhead(head)
        seqr, msg = infer_sequencer_with_message(iid, fcid)
        for sr in seqr:
            signal = (sr, msg)
        seqrs.add(signal)

    print(seqrs)


def sequencer_detection_message(fastq_files):
    seqrs = set()
    # accumulate (sequencer, status) set
    for fastq in fastq_files:
        with gzip.open(fastq) as f:
            head = str(f.readline())
            # line = str(f.readline()
            # if len(line) > 0:
            #     if line[0] == "@":
            #         head = line
            #     else:
            #         print("Incorrectly formatted first read in FASTQ file: %s" % fastq)
            #         print(line)

        iid, fcid = parse_readhead(head)
        seqr, msg = infer_sequencer_with_message(iid, fcid)
        for sr in seqr:
            signal = (sr, msg)
        seqrs.add(signal)

    # get a list of sequencing platforms
    platforms = set()
    for platform, _ in seqrs:
        platforms.add(platform)
    sequencers = list(platforms)

    # if no sequencer detected at all
    message = ""
    fails = 0
    for platform, status in seqrs:
        if status == fail_msg:
            fails += 1
    if fails == len(seqrs):
        message = "could not detect the sequencing platform(s) used to generate the input FASTQ files"
        return message, sequencers

    # if partial or no detection failures
    if fails > 0:
        message = "could not detect the sequencing platform used to generate some of the input FASTQ files, "
    message += "detected the following sequencing platforms- "
    for platform, status in seqrs:
        if status != fail_msg:
            message += platform + " " + status + ", "
    message = message.strip(", ")
    return message, sequencers

### make tree view of files
def list_files(startpath, maxlevel):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(startpath, "").count(os.sep)
        if not level > maxlevel:
            indent = " " * 4 * (level)
            print("{}{}/".format(indent, os.path.basename(root)))
            subindent = " " * 4 * (level + 1)
            for f in dirs:
                print("{}{}".format(subindent, f))
                
                
### Download genome dict
pbm_genome_name_dict = {
    "hg38": "hsapiens_gene_ensembl",
    "hg37": "hsapiens_gene_ensembl",
    "mm10": "mmusculus_gene_ensembl",
    "dm6": "dmelanogaster_gene_ensembl",
}

pbm_host_dict = {
    "hg38": "http://www.ensembl.org",
    "hg37": "http://grch37.ensembl.org/",
    "mm10": "http://nov2020.archive.ensembl.org/",
    "dm6": "http://www.ensembl.org",
}

def download_genome_annotation(inverse_genome_dict):
    annotation_dict = {}
    for genome in inverse_genome_dict.keys():
        if os.path.exists(f"{genome}_annotation.tsv"):
            print(f"Loading cached genome annotation {genome}_annotation.tsv")
            annotation = pd.read_csv(
                f"{genome}_annotation.tsv", sep="\t", header=0, index_col=0
            )
            annotation_dict[genome] = annotation
        else:
            dataset = pbm.Dataset(
                name=pbm_genome_name_dict[genome], host=pbm_host_dict[genome]
            )

            annotation = dataset.query(
                attributes=[
                    "chromosome_name",
                    "transcription_start_site",
                    "strand",
                    "external_gene_name",
                    "transcript_biotype",
                ]
            )
            filter = annotation["Chromosome/scaffold name"].str.contains("CHR|GL|JH|MT")
            annotation = annotation[~filter]
            annotation["Chromosome/scaffold name"] = annotation[
                "Chromosome/scaffold name"
            ].str.replace(r"(\b\S)", r"chr\1")
            annotation.columns = [
                "Chromosome",
                "Start",
                "Strand",
                "Gene",
                "Transcript_type",
            ]
            annotation = annotation[annotation.Transcript_type == "protein_coding"]
            annotation.to_csv(f"{genome}_annotation.tsv", sep="\t")

            annotation_dict[genome] = annotation
            
        return annotation_dict
    
### Otsu filtering
def histogram(array, nbins=100):
    """
    Draw histogram from distribution and identify centers.
    Parameters
    ---------
    array: `class::np.array`
            Scores distribution
    nbins: int
            Number of bins to use in the histogram
    Return
    ---------
    float
            Histogram values and bin centers.
    """
    array = array.ravel().flatten()
    hist, bin_edges = np.histogram(array, bins=nbins, range=None)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    return hist, bin_centers


def threshold_otsu(array, nbins=100, min_value=100):
    """
    Apply Otsu threshold on topic-region distributions [Otsu, 1979].
    Parameters
    ---------
    array: `class::np.array`
            Array containing the region values for the topic to be binarized.
    nbins: int
            Number of bins to use in the binarization histogram
    Return
    ---------
    float
            Binarization threshold.
    Reference
    ---------
    Otsu, N., 1979. A threshold selection method from gray-level histograms. IEEE transactions on systems, man, and
    cybernetics, 9(1), pp.62-66.
    """
    array = array[(array >= min_value)]
    hist, bin_centers = histogram(array, nbins)
    hist = hist.astype(float)
    # Class probabilities for all possible thresholds
    weight1 = np.cumsum(hist)
    weight2 = np.cumsum(hist[::-1])[::-1]
    # Class means for all possible thresholds
    mean1 = np.cumsum(hist * bin_centers) / weight1
    mean2 = (np.cumsum((hist * bin_centers)[::-1]) / weight2[::-1])[::-1]
    # Clip ends to align class 1 and class 2 variables:
    # The last value of ``weight1``/``mean1`` should pair with zero values in
    # ``weight2``/``mean2``, which do not exist.
    variance12 = weight1[:-1] * weight2[1:] * (mean1[:-1] - mean2[1:]) ** 2
    idx = np.argmax(variance12)
    threshold = bin_centers[:-1][idx]
    return threshold


# def calc_kde(xy):
#     return gaussian_kde(xy)(xy)


def plot_frag_qc(
    x,
    y,
    ax,
    x_thr_min=None,
    x_thr_max=None,
    y_thr_min=None,
    y_thr_max=None,
    ylab=None,
    xlab="Number of (unique) fragments in regions",
    cmap="viridis",
    density_overlay=False,
    s=10,
    marker="+",
    c="#343434",
    xlim=None,
    ylim=None,
    **kwargs,
):
    from scipy.stats import gaussian_kde

    assert all(x.index == y.index)
    barcodes = x.index.values

    if density_overlay:
        cores = 8

        x_log = np.log(x)

        # Split input array for KDE [log(x), y] array in
        # equaly spaced parts (start_offset + n * nbr_cores).
        kde_parts = [np.vstack([x_log[i::cores], y[i::cores]]) for i in range(cores)]

        # Get nultiprocess context object to spawn processes.
        # mp_ctx = mp.get_context("spawn")

        # Calculate KDE in parallel.
        with Pool(processes=cores) as pool:
            results = pool.map(kde.calc_kde, kde_parts)

        z = np.concatenate(results)

        # now order x and y in the same way that z was ordered, otherwise random z value is assigned to barcode:
        x_ordered = np.concatenate([x[i::cores] for i in range(cores)])
        y_ordered = np.concatenate([y[i::cores] for i in range(cores)])

        idx = (
            z.argsort()
        )  # order based on z value so that highest value is plotted on top, and not hidden by lower values
        x, y, z, barcodes = x_ordered[idx], y_ordered[idx], z[idx], barcodes[idx]
    else:
        z = c

    sp = ax.scatter(x, y, c=z, s=s, edgecolors=None, marker=marker, cmap=cmap, **kwargs)
    if ylim is not None:
        ax.set_ylim(ylim[0], ylim[1])
    if xlim is not None:
        ax.set_xlim(xlim[0], xlim[1])

    # Start with keeping all barcodes.
    barcodes_to_keep = np.full(x.shape[0], True)

    # Filter barcodes out if needed based on thresholds:
    if x_thr_min is not None:
        ax.axvline(x=x_thr_min, color="r", linestyle="--")

    if x_thr_max is not None:
        ax.axvline(x=x_thr_max, color="r", linestyle="--")

    if y_thr_min is not None:
        ax.axhline(y=y_thr_min, color="r", linestyle="--")

    if y_thr_max is not None:
        ax.axhline(y=y_thr_max, color="r", linestyle="--")

    ax.set_xscale("log")
    ax.set_xmargin(0.01)
    ax.set_ymargin(0.01)
    ax.set_xlabel(xlab, fontsize=10)
    ax.set_ylabel(ylab, fontsize=10)


def plot_qc(
    sample,
    sample_alias,
    min_dict,
    max_dict,
    metadata_bc_df,
    include_kde=False,
    detailed_title=True,
    s=4,
    min_x_val=100,
    min_y_val=1,
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=150)

    # calculate thresholds using a double otsu strategy:
    # first we calculate an otsu threshold using a minimum of 10 fragments
    # then, this otsu threshold is used as the minimum for the second iteration

    x_arr = np.log10(metadata_bc_df["Unique_nr_frag_in_regions"])
    x_threshold_log = threshold_otsu(x_arr, nbins=5000, min_value=np.log10(min_x_val))
    x_threshold = 10**x_threshold_log

    y_arr = metadata_bc_df["TSS_enrichment"]
    y_threshold = threshold_otsu(y_arr, nbins=5000, min_value=min_y_val)

    # calculate cells passing filter
    metadata_bc_df_passing_filters = metadata_bc_df.loc[
        (metadata_bc_df.Unique_nr_frag_in_regions > x_threshold)
        & (metadata_bc_df.TSS_enrichment > y_threshold)
    ]
    bc_passing_filters = metadata_bc_df_passing_filters.index

    # plot everything
    plot_frag_qc(
        x=metadata_bc_df["Unique_nr_frag_in_regions"],
        y=metadata_bc_df["TSS_enrichment"],
        ylab="TSS Enrichment",
        s=s,
        x_thr_min=x_threshold,
        y_thr_min=y_threshold,
        xlim=[10, max_dict["Unique_nr_frag_in_regions"]],
        ylim=[0, max_dict["TSS_enrichment"]],
        density_overlay=include_kde,
        ax=ax1,
    )
    plot_frag_qc(
        x=metadata_bc_df["Unique_nr_frag_in_regions"],
        y=metadata_bc_df["FRIP"],
        x_thr_min=x_threshold,
        ylab="FRIP",
        s=s,
        xlim=[10, max_dict["Unique_nr_frag_in_regions"]],
        ylim=[0, 1],
        density_overlay=include_kde,
        ax=ax2,
    )
    plot_frag_qc(
        x=metadata_bc_df["Unique_nr_frag_in_regions"],
        y=metadata_bc_df["Dupl_rate"],
        x_thr_min=x_threshold,
        ylab="Duplicate rate per cell",
        s=s,
        xlim=[10, max_dict["Unique_nr_frag_in_regions"]],
        ylim=[0, 1],
        density_overlay=include_kde,
        ax=ax3,
    )

    if detailed_title:
        med_nf = round(
            metadata_bc_df.loc[
                bc_passing_filters, "Unique_nr_frag_in_regions"
            ].median(),
            2,
        )
        med_tss = round(
            metadata_bc_df.loc[bc_passing_filters, "TSS_enrichment"].median(), 2
        )
        med_frip = round(metadata_bc_df.loc[bc_passing_filters, "FRIP"].median(), 2)
        title = f"{sample_alias}: Kept {len(bc_passing_filters)} cells using Otsu filtering. Median Unique Fragments: {med_nf:.0f}. Median TSS Enrichment: {med_tss:.2f}. Median FRIP: {med_frip:.2f}\nUsed a minimum of {x_threshold:.2f} fragments and TSS enrichment of {y_threshold:.2f})"
    else:
        title = sample

    fig.suptitle(title, x=0.5, y=0.95, fontsize=10)
    return bc_passing_filters, fig

### Saturation analysis
def read_bc_and_counts_from_fragments_file(fragments_bed_filename: str) -> pl.DataFrame:
    """
    Read cell barcode (column 4) and counts per fragment (column 5) from fragments BED file.
    Cell barcodes will appear more than once as they have counts per fragment, but as
    the fragment locations are not needed, they are not returned.

    Parameters
    ----------
    fragments_bed_filename: Fragments BED filename.

    Returns
    -------
    Polars dataframe with cell barcode and count per fragment (column 4 and 5 of BED file).
    """

    bed_column_names = (
        "Chromosome",
        "Start",
        "End",
        "Name",
        "Score",
        "Strand",
        "ThickStart",
        "ThickEnd",
        "ItemRGB",
        "BlockCount",
        "BlockSizes",
        "BlockStarts",
    )

    # Set the correct open function depending if the fragments BED file is gzip compressed or not.
    open_fn = gzip.open if fragments_bed_filename.endswith(".gz") else open

    skip_rows = 0
    nbr_columns = 0

    with open_fn(fragments_bed_filename, "rt") as fragments_bed_fh:
        for line in fragments_bed_fh:
            # Remove newlines and spaces.
            line = line.strip()

            if not line or line.startswith("#"):
                # Count number of empty lines and lines which start with a comment before the actual data.
                skip_rows += 1
            else:
                # Get number of columns from the first real BED entry.
                nbr_columns = len(line.split("\t"))

                # Stop reading the BED file.
                break

    if nbr_columns < 5:
        raise ValueError(
            f'Fragments BED file needs to have at least 5 columns. "{fragments_bed_filename}" contains only '
            f"{nbr_columns} columns."
        )

    # Read cell barcode (column 4) and counts (column 5) per fragemnt from fragments BED file.
    fragments_df = pl.read_csv(
        fragments_bed_filename,
        has_header=False,
        skip_rows=skip_rows,
        sep="\t",
        use_pyarrow=False,
        n_threads=6,
        columns=["column_1", "column_2", "column_3", "column_4", "column_5"],
        new_columns=["Chromosome", "Start", "End", "CellBarcode", "FragmentCount"],
        dtypes=[pl.Categorical, pl.UInt32, pl.UInt32, pl.Categorical, pl.UInt32],
    )

    return fragments_df

sampling_fractions_default = [
    0.0,
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.55,
    0.6,
    0.65,
    0.7,
    0.75,
    0.8,
    0.85,
    0.9,
    0.92,
    0.94,
    0.96,
    0.98,
    1.0,
]


def sub_sample_fragments(
    fragments_df,
    min_uniq_frag=200,
    selected_barcodes=[],
    sampling_fractions=sampling_fractions_default,
    stats_tsv_filename="sampling_stats.tsv",
    whitelist=None,
):
    sampling_fractions_length = len(sampling_fractions)

    # Initialize dataframe for storing all statistics results.
    stats_df = pd.DataFrame(
        {
            "mean_frag_per_bc": np.zeros(sampling_fractions_length, np.float64),
            "median_uniq_frag_per_bc": np.zeros(sampling_fractions_length, np.float64),
            "total_unique_frag_count": np.zeros(sampling_fractions_length, np.uint32),
            "total_frag_count": np.zeros(sampling_fractions_length, np.uint32),
            "cell_barcode_count": np.zeros(sampling_fractions_length, np.uint32),
        },
        index=pd.Index(data=np.array(sampling_fractions), name="sampling_fraction"),
    )

    # Get all cell barcodes which have more than min_uniq_frag fragments.
    if not selected_barcodes:
        selected_barcodes = (
            fragments_df.groupby("CellBarcode")
            .agg(pl.col("FragmentCount").count().alias("nbr_frags_per_CBs"))
            .filter(pl.col("nbr_frags_per_CBs") > min_uniq_frag)
        )
    else:
        selected_barcodes = pl.DataFrame(
            [
                pl.Series("CellBarcode", selected_barcodes, dtype=pl.Categorical),
            ]
        )

    # Count all good cell barcodes.
    nbr_selected_barcodes = selected_barcodes.height

    if 1.0 in sampling_fractions:
        # As there is no need to sample when sampling fraction is 100%,
        # the median number of unique fragments per barcode can be
        # calculated much more efficiently on the original fragments
        # file dataframe with counts than the expanded one, which is
        # needed when sampling is required.

        print("Calculate statistics for sampling fraction 100.0%.")

        print(f"Keep fragments with good barcodes.")
        fragments_for_good_bc_df = selected_barcodes.join(
            fragments_df, left_on="CellBarcode", right_on="CellBarcode", how="left"
        )

        print("Calculate total number of fragments.")

        stats_df.loc[1.0, "total_unique_frag_count"] = (
            fragments_for_good_bc_df.groupby(
                ["CellBarcode", "Chromosome", "Start", "End"]
            )
            .agg([pl.first("Start").alias("Start_tmp")])
            .select(pl.count())
            .item()
        )

        stats_df.loc[1.0, "total_frag_count"] = fragments_for_good_bc_df.select(
            pl.col("FragmentCount").sum()
        ).item()

        print(
            "Calculate mean number of fragments per barcode and median number of unique"
            " fragments per barcode."
        )
        stats_df_pl = (
            fragments_for_good_bc_df.groupby("CellBarcode")
            .agg(
                [
                    pl.col("FragmentCount").sum().alias("MeanFragmentsPerCB"),
                    pl.count().alias("UniqueFragmentsPerCB"),
                ]
            )
            .select(
                [
                    pl.col("MeanFragmentsPerCB").mean(),
                    pl.col("UniqueFragmentsPerCB").median(),
                ]
            )
        )

        stats_df.loc[1.0, "mean_frag_per_bc"] = stats_df_pl["MeanFragmentsPerCB"][0]
        stats_df.loc[1.0, "median_uniq_frag_per_bc"] = stats_df_pl[
            "UniqueFragmentsPerCB"
        ][0]
        stats_df.loc[1.0, "cell_barcode_count"] = nbr_selected_barcodes

        # Delete dataframe to free memory.
        del fragments_for_good_bc_df

    # Create dataframe where each row contains one fragment:
    #   - Original dataframe has a count per fragment with the same cell barcode.
    #   - Create a row for each count, so we can sample fairly afterwards.
    print("Create dataframe with all fragments (for sampling).")
    fragments_all_df = fragments_df.with_columns(
        pl.col("FragmentCount").repeat_by(pl.col("FragmentCount"))
    ).explode("FragmentCount")

    # Delete input dataframe to free memory.
    del fragments_df

    for sampling_fraction in sampling_fractions:
        if sampling_fraction == 0.0:
            # All statistics are zero and already set when the stats_df dataframe is created.
            continue
        elif sampling_fraction == 1.0:
            # Statistics for 100% sampling are already calculated as there is no need
            # to have the fragments_all_df dataframe as no sampling is needed.
            # This avoids the need to use the expensive groupby operations for the
            # calculations of the median number of unique fragments per barcode.
            continue

        print(
            "Calculate statistics for sampling fraction"
            f" {round(sampling_fraction * 100, 1)}%."
        )

        # Sample x% from all fragments (with duplicates) and keep fragments which have good barcodes.
        print(
            f"Sample {round(sampling_fraction * 100, 1)}% from all fragments and keep"
            " fragments with good barcodes."
        )
        fragments_sampled_for_good_bc_df = selected_barcodes.join(
            fragments_all_df.sample(frac=sampling_fraction),
            left_on="CellBarcode",
            right_on="CellBarcode",
            how="left",
        )

        # Get number of sampled fragments (with possible duplicate fragments) which have good barcodes.
        stats_df.loc[sampling_fraction, "total_unique_frag_count"] = (
            fragments_sampled_for_good_bc_df.groupby(
                ["CellBarcode", "Chromosome", "Start", "End"]
            )
            .agg([pl.first("Start").alias("Start_tmp")])
            .select(pl.count())
            .item()
        )

        stats_df.loc[
            sampling_fraction, "total_frag_count"
        ] = fragments_sampled_for_good_bc_df.select(pl.count()).item()

        print("Calculate mean number of fragments per barcode.")
        stats_df.loc[sampling_fraction, "mean_frag_per_bc"] = (
            fragments_sampled_for_good_bc_df.select(
                [pl.col("CellBarcode"), pl.col("FragmentCount")]
            )
            .groupby("CellBarcode")
            .agg([pl.count("FragmentCount").alias("FragmentsPerCB")])
            .select([pl.col("FragmentsPerCB").mean().alias("MeanFragmentsPerCB")])[
                "MeanFragmentsPerCB"
            ][0]
        )

        print("Calculate median number of unique fragments per barcode.")
        stats_df.loc[sampling_fraction, "median_uniq_frag_per_bc"] = (
            fragments_sampled_for_good_bc_df.groupby(
                ["CellBarcode", "Chromosome", "Start", "End"]
            )
            .agg([pl.col("FragmentCount").first().alias("FragmentCount")])
            .select([pl.col("CellBarcode"), pl.col("FragmentCount")])
            .groupby("CellBarcode")
            .agg(pl.col("FragmentCount").count().alias("UniqueFragmentsPerCB"))
            .select(pl.col("UniqueFragmentsPerCB").median())["UniqueFragmentsPerCB"][0]
        )

        stats_df.loc[sampling_fraction, "cell_barcode_count"] = nbr_selected_barcodes

        # Delete dataframe to free memory.
        del fragments_sampled_for_good_bc_df

    print(f'Saving statistics in "{stats_tsv_filename}".')
    stats_df.to_csv(stats_tsv_filename, sep="\t")

    return stats_df


### Plot duplication

def MM(x, Vmax, Km):
    """
    Define the Michaelis-Menten Kinetics model that will be used for the model fitting.
    """
    if Vmax > 0 and Km > 0:
        y = (Vmax * x) / (Km + x)
    else:
        y = 1e10
    return y


def plot_saturation_fragments(
    filepath,
    sample,
    n_reads,
    n_cells,
    percentage_toplot,
    svg_output_path,
    png_output_path,
    plot_current_saturation=True,
    x_axis="mean_reads_per_barcode",
    y_axis="median_uniq_frag_per_bc",
    function=MM,
):

    fig, ax = plt.subplots(figsize=(6, 4))

    stats_df = pd.read_csv(filepath, sep="\t", index_col=0)

    stats_df["total_reads"] = n_reads * stats_df.index

    stats_df["mean_reads_per_barcode"] = (
        stats_df["total_reads"] / stats_df["cell_barcode_count"]
    )
    stats_df["mean_reads_per_barcode"].fillna(0, inplace=True)
    stats_df["duplication_rate"] = (
        stats_df["total_frag_count"] - stats_df["total_unique_frag_count"]
    ) / stats_df["total_frag_count"]
    stats_df["duplication_rate"] = stats_df["duplication_rate"].fillna(0)
    # stats_df["duplication_rate"] = (stats_df["total_frag_count"] - stats_df["total_unique_frag_count"]/stats_df["total_frag_count"])
    # select x/y data fro MM fit from subsampling stats
    x_data = np.array(stats_df.loc[0:, x_axis]) / 10**3
    y_data = np.array(stats_df.loc[0:, y_axis])
    # fit to MM function

    best_fit_ab, covar = curve_fit(function, x_data, y_data, bounds=(0, +np.inf))

    # expand fit space
    x_fit = np.linspace(0, int(np.max(x_data) * 1000), num=100000)
    y_fit = function(x_fit, *(best_fit_ab))
    # impute maximum saturation to plot as 95% of y_max
    y_val = best_fit_ab[0] * 0.95
    # subset x_fit space if bigger then y_val
    if y_val < max(y_fit):
        x_coef = np.where(y_fit >= y_val)[0][0]
        x_fit = x_fit[0:x_coef]
        y_fit = y_fit[0:x_coef]

    # plot model
    ax.plot(x_fit, function(x_fit, *best_fit_ab), label="fitted", c="black", linewidth=1)
    # plot raw data
    ax.scatter(x=x_data, y=y_data, c="red", s=10)

    # mark curent saturation
    curr_x_idx = np.where(y_fit >= max(y_data))[0][0]
    curr_x_coef = max(x_data)
    curr_y_coef = max(y_data)
    if plot_current_saturation == True:
        ax.plot([curr_x_coef, curr_x_coef], [0, 9999999], linestyle="--", c="red")
        ax.plot([0, curr_x_coef], [curr_y_coef, curr_y_coef], linestyle="--", c="red")
        ax.text(
            x=curr_x_coef * 1.1,
            y=curr_y_coef * 0.9,
            s=str(round(curr_y_coef, 1))
            + " fragments, {:.2f}".format(curr_x_coef)
            + " kRPC",
            c="red",
            ha="left",
            va="bottom",
        )

    # Find read count for percent saturation
    y_val = best_fit_ab[0] * 0.9 * percentage_toplot
    # Find closest match in fit
    if max(y_fit) > y_val:
        x_idx = np.where(y_fit >= y_val)[0][0]
        x_coef = x_fit[x_idx]
        y_coef = y_fit[x_idx]
        # Draw vline
        ax.plot([x_coef, x_coef], [0, 9999999], linestyle="--", c="blue")
        # Draw hline
        ax.plot([0, x_coef], [y_coef, y_coef], linestyle="--", c="blue")
        # Plot imputed read count
        ax.text(
            x=x_coef * 1.1,
            y=y_coef * 0.9,
            s=str(round(y_coef, 1)) + " fragments, {:.2f}".format(x_coef) + " kRPC",
            c="blue",
            ha="left",
            va="bottom",
        )

    # get xlim value
    y_max = y_fit[-1] * 0.95
    x_idx = np.where(y_fit >= y_max)[0][0]
    x_max = x_fit[x_idx]
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, y_max])

    # add second axis
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    upper_xticklabels = [
        str(int(x)) for x in ax.get_xticks() * n_cells / 1000
    ]
    ax2.set_xticklabels(upper_xticklabels)
    ax2.set_xlabel("Total reads (millions)")
    ax2.set_xlim([0, x_max])

    # save figure
    ax.set_xlabel("Reads per cell (thousands)")

    # plt.yscale("log")
    # ax.set_xscale("log")

    ax.set_ylabel(y_axis)
    title_str = f"{sample}\n{n_cells} cells, {round(n_reads/1000000)}M reads\nCurrently at {int(curr_y_coef)} {y_axis} with {int(curr_x_coef)} kRPC\nFor {int(percentage_toplot*100)}% saturation: {int(x_coef*n_cells/1000)}M reads needed"
    ax.set_title(title_str)

    plt.savefig(png_output_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_output_path, dpi=300, bbox_inches='tight')
    plt.show()

    plt.close()
    print(title_str)

def MM_duplication(x, Km):
    """
    Define the Michaelis-Menten Kinetics model that will be used for the model fitting.
    """
    if Km > 0:
        y = x / (Km + x)
    else:
        y = 1e10
    return y

def plot_saturation_duplication(
    filepath,
    sample,
    n_reads,
    n_cells,
    percentage_toplot,
    png_output_path,
    svg_output_path,
    function=MM_duplication,
    plot_current_saturation=True,
    x_axis="mean_reads_per_barcode",
    y_axis="median_uniq_frag_per_bc",
):
    fig, ax = plt.subplots(figsize=(6, 4))

    stats_df = pd.read_csv(filepath, sep="\t", index_col=0)
    stats_df["total_reads"] = n_reads * stats_df.index

    stats_df["mean_reads_per_barcode"] = (
        stats_df["total_reads"] / stats_df["cell_barcode_count"]
    )
    stats_df["mean_reads_per_barcode"].fillna(0, inplace=True)
    stats_df["duplication_rate"] = (
        stats_df["total_frag_count"] - stats_df["total_unique_frag_count"]
    ) / stats_df["total_frag_count"]
    stats_df["duplication_rate"] = stats_df["duplication_rate"].fillna(0)
    # stats_df["duplication_rate"] = (stats_df["total_frag_count"] - stats_df["total_unique_frag_count"]/stats_df["total_frag_count"])
    # select x/y data fro MM fit from subsampling stats
    x_data = np.array(stats_df.loc[0:, x_axis]) / 10**3
    y_data = np.array(stats_df.loc[0:, y_axis])
    # fit to MM function

    best_fit_ab, covar = curve_fit(function, x_data, y_data, bounds=(0, +np.inf))

    # expand fit space
    x_fit = np.linspace(0, int(np.max(x_data) * 1000), num=100000)
    y_fit = function(x_fit, *(best_fit_ab))
    # impute maximum saturation to plot as 95% of y_max
    y_val = best_fit_ab[0] * 0.95
    # subset x_fit space if bigger then y_val
    if y_val < max(y_fit):
        x_coef = np.where(y_fit >= y_val)[0][0]
        x_fit = x_fit[0:x_coef]
        y_fit = y_fit[0:x_coef]

    # plot model
    ax.plot(
        x_fit,
        function(x_fit, *best_fit_ab),
        label="fitted",
        c="black",
        linewidth=1,
    )
    # plot raw data
    ax.scatter(x=x_data, y=y_data, c="red", s=10)

    # mark curent saturation
    curr_x_idx = np.where(y_fit >= max(y_data))[0][0]
    curr_x_coef = max(x_data)
    curr_y_coef = max(y_data)
    if plot_current_saturation == True:
        ax.plot([curr_x_coef, curr_x_coef], [0, 99999], linestyle="--", c="r")
        ax.plot([0, curr_x_coef], [curr_y_coef, curr_y_coef], linestyle="--", c="r")
        ax.text(
            x=curr_x_coef * 1.1,
            y=curr_y_coef * 0.9,
            s=str(round(100 * curr_y_coef)) + "% {:.2f}".format(curr_x_coef) + " kRPC",
            c="r",
            ha="left",
            va="bottom",
        )

    # Find read count for percent saturation
    y_val = percentage_toplot
    # Find closest match in fit
    if max(y_fit) > y_val:
        x_idx = np.where(y_fit >= y_val)[0][0]
        x_coef = x_fit[x_idx]
        y_coef = y_fit[x_idx]
        # Draw vline
        ax.plot([x_coef, x_coef], [0, 99999], linestyle="--", c="blue")
        # Draw hline
        ax.plot([0, x_coef], [y_coef, y_coef], linestyle="--", c="blue")
        # Plot imputed read count
        ax.text(
            x=x_coef * 1.1,
            y=y_coef * 0.9,
            s=str(round(100 * percentage_toplot))
            + "% @ {:.2f}".format(x_coef)
            + " kRPC",
            c="blue",
            ha="left",
            va="bottom",
        )

    # get xlim value
    y_max = 0.9
    x_idx = np.where(y_fit >= y_max)[0][0]
    x_max = x_fit[x_idx]
    ax.set_xlim([0, x_max])
    ax.set_ylim([0, 1])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
    # add second axis
    ax2 = ax.twiny()
    ax2.set_xticks(ax.get_xticks())
    upper_xticklabels = [
        str(int(x)) for x in ax.get_xticks() * n_cells / 1000
    ]
    ax2.set_xticklabels(upper_xticklabels)
    ax2.set_xlabel("Total reads (millions)")
    ax2.set_xlim([0, x_max])

    ax.set_xlabel("Reads per cell (thousands)")
    # plt.yscale("log")
    # ax.set_xscale("log")

    ax.set_ylabel("Duplication rate (%)")
    title_str = f"{sample}\n{n_cells} cells, {round(n_reads/1000000)}M reads\nCurrently at {int(curr_y_coef*100)}% duplication rate with {int(curr_x_coef)} kRPC\nFor {int(percentage_toplot*100)}% duplication rate, {int(x_coef*n_cells/1000)}M reads needed"
    ax.set_title(title_str)

    plt.savefig(png_output_path, dpi=300, bbox_inches='tight')
    plt.savefig(svg_output_path, dpi=300, bbox_inches='tight')
    plt.show()

    plt.close()
    print(title_str)
    
### Parsing data
def scrape_mapping_stats(samples, samples_tech_dict, pipeline, output_dir, verbose):
    df_stats = pd.DataFrame(index=pd.Index(samples))
    tsv_list = sorted(glob.glob(f"selected_barcodes/*otsu.txt"))
    cell_count_dict = {}
    for tsv in tsv_list:
        sample = tsv.split("/")[-1].split("_bc_passing_filters_otsu")[0]
        # print(sample)
        df = pd.read_csv(tsv, sep="\t", index_col=0)
        cell_count_dict[sample] = len(df)
        df_stats.loc[sample, "sample_id"] = sample

        df_stats.loc[sample, "n_cells"] = len(df)

    if pipeline == "PUMATAC":
        directory = f"{output_dir}/data/reports/barcode/"
        for sample in df_stats.index:
            file = glob.glob(f"{directory}/*{sample}*.corrected.bc_stats.log")[0]
            if os.path.exists(file):
                # print(f"{sample}: {file}")
                df = pd.read_csv(
                    file, sep="\t\t|\t", engine="python", index_col=0, header=None
                )
                # print(df)
                tech = samples_tech_dict[sample]
                if tech == "biorad":
                    nreads = df.loc["nbr_reads:", 1]
                    nbarcodes_total = df.loc[
                        "nbr_reads_with_bc1_bc2_bc3_correct_or_correctable", 1
                    ]
                    percentage_correct_barcodes = nbarcodes_total / nreads * 100
                else:
                    nreads = df.loc["nbr_reads:", 1]
                    nbarcodes_total = df.loc["total_bc_found", 1]
                    percentage_correct_barcodes = nbarcodes_total / nreads * 100

                if verbose == True:
                    print(f"nreads: {nreads}")
                    print(f"nbarcodes_total: {nbarcodes_total}")
                    print(f"percentage_correct_barcodes: {percentage_correct_barcodes}")
                    print("-------------------------------------\n")

                df_stats.loc[sample, "n_reads"] = int(nreads)
                df_stats.loc[sample, "%_correct_barcodes"] = round(
                    percentage_correct_barcodes, 2
                )
            else:
                print(f"{file} does not exist!")

        directory = f"{output_dir}/data/reports/mapping_stats/"
        for sample in df_stats.index:
            file = directory + sample + "_____R1.mapping_stats.tsv"
            if os.path.exists(file):
                print(f"{sample}: {file}")
                df = pd.read_csv(file, sep="\t", engine="python", index_col=0, header=0)
                if verbose == True:
                    print(df.astype(int))
                    print("\n")

                percent_mapq30 = (
                    df.loc["Reads mapped with MAPQ>30:"]
                    / df.loc["raw total sequences:"]
                    * 100
                )
                avg_insert = df.loc["insert size average:"]
                avg_map_quality = df.loc["average quality:"]
                r1_length = df.loc["maximum first fragment length:"]
                r2_length = df.loc["maximum last fragment length:"]

                if verbose == True:

                    print(f"read 1 length: {int(r1_length)}")
                    print(f"read 2 length: {int(r2_length)}")
                    print(f"average map quality: {round(avg_map_quality, 2)}")
                    print(f"percent mapq30: {round(percent_mapq30, 2)}")
                    print(f"insert size average: {avg_insert}")
                    print("-------------------------------------\n")

                df_stats.loc[sample, "r1_length"] = int(r1_length)
                df_stats.loc[sample, "r2_length"] = int(r2_length)
                df_stats.loc[sample, "avg_insert_size"] = int(avg_insert)
                df_stats.loc[sample, "%_mapq30"] = round(percent_mapq30.iloc[0], 2)
                df_stats.loc[sample, "avg_map_quality"] = round(
                    avg_map_quality.iloc[0], 2
                )
            elif verbose == True:
                print(f"{file}")

    elif pipeline == "cellranger-arc" or pipeline == "cellranger-atac":
        stats_dict = {
            x.split("/")[1]: x
            for x in sorted(glob.glob(f"{output_dir}/*/outs/summary.csv"))
        }

        for sample, filepath in stats_dict.items():
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                percentage_correct_barcodes = df["ATAC Valid barcodes"][0] * 100
                n_reads = df["ATAC Sequenced read pairs"][0]
                percent_mapq30 = df["ATAC Confidently mapped read pairs"][0] * 100

                if verbose == True:
                    print(f"percentage_correct_barcodes: {percentage_correct_barcodes}")
                    print(f"percent mapq30: {round(percent_mapq30, 2)}")
                    print("-------------------------------------\n")

                df_stats.loc[sample, "%_correct_barcodes"] = round(
                    percentage_correct_barcodes, 2
                )
                df_stats.loc[sample, "n_reads"] = n_reads

                df_stats.loc[sample, "%_mapq30"] = round(percent_mapq30, 2)

            else:
                print(f"{file} does not exist!")

    return df_stats

def scrape_scstats(metadata_path_dict, selected_cells_path_dict, df_stats):
    df_merged = pd.DataFrame()
    df_scstats_merged = pd.DataFrame()
    for sample in metadata_path_dict.keys():
        with open(metadata_path_dict[sample], "rb") as f:
            df = pickle.load(f)

        with open(selected_cells_path_dict[sample], "rb") as f:
            selected_barcodes = pickle.load(f)

        df = df.loc[selected_barcodes]
        df_median = df.median()
        df_median.index = ["Median_" + x.lower() for x in df_median.index]
        df_median["total_nr_frag_in_selected_barcodes"] = sum(df["Total_nr_frag"])
        df_median["total_nr_unique_frag_in_selected_barcodes"] = sum(
            df["Unique_nr_frag"]
        )
        df_median["total_nr_unique_frag_in_selected_barcodes_in_regions"] = sum(
            df["Unique_nr_frag_in_regions"]
        )
        df_median["n_barcodes_merged"] = len(
            [x for x in [x.split("__")[0] for x in df.index] if "_" in x]
        )
        df_median["frac_barcodes_merged"] = len(
            [x for x in [x.split("__")[0] for x in df.index] if "_" in x]
        ) / len(df)
        df_merged = pd.concat([df_merged, df_median], axis=1)
        df_scstats_merged = pd.concat([df_scstats_merged, df], axis=0)

    df_merged.columns = metadata_path_dict.keys()

    df_merged = df_merged.T
    df_merged = df_merged[
        [
            "Median_total_nr_frag",
            "Median_unique_nr_frag",
            "Median_dupl_rate",
            "Median_total_nr_frag_in_regions",
            "Median_frip",
            "Median_tss_enrichment",
            "total_nr_frag_in_selected_barcodes",
            "total_nr_unique_frag_in_selected_barcodes",
            "total_nr_unique_frag_in_selected_barcodes_in_regions",
            "n_barcodes_merged",
            "frac_barcodes_merged",
        ]
    ]

    df_scstats_merged["tech"] = "user_sample"
    df_merged["tech"] = "user_sample"

    df_scstats_merged_benchmark = pd.read_csv(
        "PUMATAC_dependencies/data/fixedcells_cto_merged.tsv",
        sep="\t",
    )

    df_scstats_merged_benchmark.index = df_scstats_merged_benchmark["Unnamed: 0"]
    df_scstats_merged_benchmark.drop(
        ["Unnamed: 0.1", "Unnamed: 0"], inplace=True, axis=1
    )
    df_scstats_merged_benchmark.index = [
        x.replace("CNA_10xv11_4", "CNA_10xv11c_1")
        .replace("CNA_10xv11_5", "CNA_10xv11c_2")
        .replace("BRO_mtscatac", "BRO_mtscatacfacs")
        for x in df_scstats_merged_benchmark.index
    ]
    df_scstats_merged_benchmark["sample_id"] = [
        x.replace("CNA_10xv11_4", "CNA_10xv11c_1")
        .replace("CNA_10xv11_5", "CNA_10xv11c_2")
        .replace("BRO_mtscatac", "BRO_mtscatacfacs")
        for x in df_scstats_merged_benchmark["sample_id"]
    ]
    df_scstats_merged_benchmark["tech"] = [
        x.split("___")[-1].split("_")[1] for x in df_scstats_merged_benchmark.index
    ]

    df_scstats_merged_benchmark = df_scstats_merged_benchmark[df_scstats_merged.columns]

    df_scstats_merged = pd.concat([df_scstats_merged_benchmark, df_scstats_merged])

    df_scstats_merged["Unique_nr_frag_in_regions_k"] = (
        df_scstats_merged["Unique_nr_frag_in_regions"] / 1000
    )

    df_stats = pd.concat([df_stats, df_merged], axis=1)
    return df_scstats_merged, df_stats

### Calculate losses
def calculate_losses(df_stats, df_scstats_merged):
    for user_sample in df_stats.index:
        df_stats.at[user_sample, "total_nr_frag_in_selected_barcodes"] = (
            df_scstats_merged.groupby("sample_id")["Total_nr_frag"]
            .sum()
            .loc[user_sample]
        )
        df_stats.at[user_sample, "total_nr_unique_frag_in_selected_barcodes"] = (
            df_scstats_merged.groupby("sample_id")["Unique_nr_frag"]
            .sum()
            .loc[user_sample]
        )
        df_stats.at[
            user_sample, "total_nr_unique_frag_in_selected_barcodes_in_regions"
        ] = (
            df_scstats_merged.groupby("sample_id")["Unique_nr_frag_in_regions"]
            .sum()
            .loc[user_sample]
        )

    df_sub = df_stats[
        [
            "n_reads",
            "total_nr_frag_in_selected_barcodes",
            "total_nr_unique_frag_in_selected_barcodes",
            "total_nr_unique_frag_in_selected_barcodes_in_regions",
        ]
    ]

    df_sub["with_correct_barcode"] = (
        df_sub["n_reads"] * df_stats["%_correct_barcodes"] / 100
    )

    df_sub["mapped"] = df_sub["with_correct_barcode"] * df_stats["%_mapq30"] / 100

    df_sub = df_sub.div(df_sub["n_reads"], axis=0)

    df_sub["No correct barcode"] = df_sub["n_reads"] - df_sub["with_correct_barcode"]
    df_sub["Not mapped properly"] = df_sub["with_correct_barcode"] - df_sub["mapped"]

    df_sub["Fragments in background noise barcodes"] = (
        df_sub["mapped"] - df_sub["total_nr_frag_in_selected_barcodes"]
    )
    df_sub["Duplicate fragments in cells"] = (
        df_sub["total_nr_frag_in_selected_barcodes"]
        - df_sub["total_nr_unique_frag_in_selected_barcodes"]
    )
    df_sub["Unique fragments in cells, not in peaks"] = (
        df_sub["total_nr_unique_frag_in_selected_barcodes"]
        - df_sub["total_nr_unique_frag_in_selected_barcodes_in_regions"]
    )
    df_sub["Unique fragments in cells and in peaks"] = df_sub[
        "total_nr_unique_frag_in_selected_barcodes_in_regions"
    ]

    # df_sub.columns = [
    #     "No correct barcode",
    #     "Duplicate fragments in cells",
    #     "Unique fragments in cells, not in peaks",
    #     "Unique fragments in cells and in peaks",
    #     "Not mapped properly",
    #     "Fragments in background noise barcodes",
    # ]

    df_sub = df_sub[
        [
            "No correct barcode",
            "Not mapped properly",
            "Fragments in background noise barcodes",
            "Duplicate fragments in cells",
            "Unique fragments in cells, not in peaks",
            "Unique fragments in cells and in peaks",
        ]
    ]
    if not "No correct barcode" in df_stats.columns:
        df_stats = pd.concat([df_stats, df_sub], axis=1)

    df_stats_reference = pd.read_csv(
        "PUMATAC_dependencies/data/fixedcells_general_statistics.tsv",
        sep="\t",
        index_col=0,
    )
    df_stats_reference["tech"] = df_stats_reference["technology"]
    df_stats_reference["sample_id"] = df_stats_reference.index

    df_stats_reference["n_cells"] = df_stats_reference["cells"]
    df_stats_reference["n_reads"] = df_stats_reference["reads"]
    df_stats_reference["with_correct_barcode"] = (
        df_stats_reference["n_reads"] * df_stats_reference["%_correct_barcodes"] / 100
    )
    df_stats_reference["mapped"] = (
        df_stats_reference["with_correct_barcode"]
        * df_stats_reference["%_mapq30"]
        / 100
    )
    df_stats_reference = df_stats_reference[df_stats.columns]
    df_stats_reference.index = [x + ".FIXEDCELLS" for x in df_stats_reference.index]
    df_stats_reference["tech"] = [x.split("_")[1] for x in df_stats_reference.index]

    df_stats["tech"] = "user_sample"
    df_stats_merged = pd.concat([df_stats_reference, df_stats])
    rename_dict = {
        "Fragments in background noise barcodes": "Fragments in noise barcodes",
        "Unique fragments in cells, not in peaks": "Unique, in cells, not in peaks",
        "Unique fragments in cells and in peaks": "Unique, in cells, in peaks",
    }
    df_stats_merged = df_stats_merged.rename(columns=rename_dict)

    return df_stats_merged


### barplots
def plot_all_qc(
    df_stats_merged,
    df_scstats_merged,
    variables_list,
    sample_order,
    sample_alias_dict,
    tech_order,
    ylim_dict,
    svg_output_path,
    png_output_path,
    individual_barplot_width=0.5,
    individual_plot_row_height=4,
):
    ### Initialize some objects
    tech_color_palette = {
        "10xv2": "#1b9e77",
        "10xv1": "#d95f02",
        "10xv11": "#7570b3",
        "10xv11c": "#7570b3",
        "10xmultiome": "#e7298a",
        "mtscatac": "#66a61e",
        "mtscatacfacs": "#66a61e",
        "ddseq": "#e6ab02",
        "s3atac": "#a6761d",
        "hydrop": "#666666",
        "user_sample": "#FF0000",
    }

    var_alias_dict = {
        "Log_total_nr_frag": "Total Fragments",
        "Log_unique_nr_frag": "Total Fragments",
        "Total_nr_frag": "Total Fragments",
        "Unique_nr_frag": "Unique Fragments",
        "Dupl_nr_frag": "Duplicate Fragments",
        "Dupl_rate": "% Duplicate Fragments",
        "Total_nr_frag_in_regions": "Total Fragments in Regions",
        "Unique_nr_frag_in_regions": "Unique Fragments\nin Peaks",
        "Unique_nr_frag_in_regions_k": "Unique Fragments\nin Peaks (x1000)",
        "FRIP": "Fraction of Unique\nFragments in Peaks",
        "TSS_enrichment": "TSS\nEnrichment",
        "sample_id": "Sample",
        "tech": "Technology",
        "seurat_cell_type_pred_score": "Seurat score",
        "Doublet_scores_fragments": "Scrublet score",
    }

    order_dict = {
        "10xmultiome": [
            "SAN_10xmultiome_1.FIXEDCELLS",
            "SAN_10xmultiome_2.FIXEDCELLS",
            "CNA_10xmultiome_1.FIXEDCELLS",
            "CNA_10xmultiome_2.FIXEDCELLS",
            "VIB_10xmultiome_2.FIXEDCELLS",
            "VIB_10xmultiome_1.FIXEDCELLS",
        ],
        "10xv1": ["VIB_10xv1_1.FIXEDCELLS", "VIB_10xv1_2.FIXEDCELLS"],
        "10xv11": [
            "TXG_10xv11_1.FIXEDCELLS",
            "CNA_10xv11_3.FIXEDCELLS",
            "CNA_10xv11_2.FIXEDCELLS",
            "CNA_10xv11_1.FIXEDCELLS",
            "STA_10xv11_1.FIXEDCELLS",
            "STA_10xv11_2.FIXEDCELLS",
        ],
        "10xv11c": ["CNA_10xv11c_1.FIXEDCELLS", "CNA_10xv11c_2.FIXEDCELLS"],
        "10xv2": [
            "VIB_10xv2_2.FIXEDCELLS",
            "VIB_10xv2_1.FIXEDCELLS",
            "TXG_10xv2_1.FIXEDCELLS",
            "TXG_10xv2_2.FIXEDCELLS",
            "CNA_10xv2_1.FIXEDCELLS",
            "CNA_10xv2_2.FIXEDCELLS",
        ],
        "ddseq": [
            "HAR_ddseq_1.FIXEDCELLS",
            "HAR_ddseq_2.FIXEDCELLS",
            "BIO_ddseq_2.FIXEDCELLS",
            "BIO_ddseq_4.FIXEDCELLS",
            "BIO_ddseq_1.FIXEDCELLS",
            "BIO_ddseq_3.FIXEDCELLS",
            "UCS_ddseq_2.FIXEDCELLS",
            "UCS_ddseq_1.FIXEDCELLS",
        ],
        "hydrop": [
            "EPF_hydrop_4.FIXEDCELLS",
            "EPF_hydrop_3.FIXEDCELLS",
            "EPF_hydrop_1.FIXEDCELLS",
            "EPF_hydrop_2.FIXEDCELLS",
            "VIB_hydrop_2.FIXEDCELLS",
            "VIB_hydrop_1.FIXEDCELLS",
            "CNA_hydrop_3.FIXEDCELLS",
            "CNA_hydrop_1.FIXEDCELLS",
            "CNA_hydrop_2.FIXEDCELLS",
        ],
        "mtscatacfacs": [
            "BRO_mtscatacfacs_1.FIXEDCELLS",
            "BRO_mtscatacfacs_2.FIXEDCELLS",
        ],
        "mtscatac": [
            "MDC_mtscatac_1.FIXEDCELLS",
            "MDC_mtscatac_2.FIXEDCELLS",
            "CNA_mtscatac_2.FIXEDCELLS",
            "CNA_mtscatac_1.FIXEDCELLS",
        ],
        "s3atac": ["OHS_s3atac_1.FIXEDCELLS", "OHS_s3atac_2.FIXEDCELLS"],
        "user_sample": sample_order,
    }

    order_dict_tech_ultrashort = {
        "10xmultiome": ["MO Sa1", "MO Sa2", "MO C1", "MO C2", "MO V2", "MO V1"],
        "10xv1": ["v1 V1", "v1 V2"],
        "10xv11": ["v1.1 T1", "v1.1 C3", "v1.1 C2", "v1.1 C1", "v1.1 St1", "v1.1 St2"],
        "10xv11c": ["v1.1c C1", "v1.1c C2"],
        "10xv2": ["v2 V2", "v2 V1", "v2 T1", "v2 T2", "v2 C1", "v2 C2"],
        "ddseq": [
            "ddS H1",
            "ddS H2",
            "ddS Bi2",
            "ddS Bi4",
            "ddS Bi1",
            "ddS Bi3",
            "ddS U2",
            "ddS U1",
        ],
        "hydrop": [
            "Hy E4",
            "Hy E3",
            "Hy E1",
            "Hy E2",
            "Hy V2",
            "Hy V1",
            "Hy C3",
            "Hy C1",
            "Hy C2",
        ],
        "mtscatacfacs": ["mt* Br1", "mt* Br2"],
        "mtscatac": ["mt M1", "mt M2", "mt C2", "mt C1"],
        "s3atac": ["s3 O1", "s3 O2"],
        "user_sample": [sample_alias_dict[x] for x in sample_order],
    }


    losses_order = [
        "tech",
        "No correct barcode",
        "Not mapped properly",
        "Fragments in noise barcodes",
        "Duplicate fragments in cells",
        "Unique, in cells, not in peaks",
        "Unique, in cells, in peaks",
    ]

    losses_order = losses_order[::-1]

    losses_color_palette = palettable.cartocolors.qualitative.Safe_7.get_mpl_colormap()

    tech_alias_dict = {
        "10xmultiome": "10x\nMultiome",
        "10xv1": "10x v1",
        "10xv11": "10x v1.1",
        "10xv11c": "10x v1.1\ncontrols",
        "10xv2": "10x v2",
        "ddseq": "Bio-Rad\nddSEQ SureCell",
        "hydrop": "HyDrop",
        "mtscatac": "mtscATAC-seq",
        "mtscatacfacs": "*",
        "s3atac": "s3-ATAC",
        "user_sample": "User samples",
    }



    ### plot
    n_samples = len(df_stats_merged["sample_id"].unique())
    n_var = len(variables_list)

    fig = plt.figure(
        figsize=(
            individual_barplot_width * (n_samples),
            individual_plot_row_height * n_var,
        ),
    )

    gs = GridSpec(
        len(variables_list) + 1,
        n_samples,
        figure=fig,
    )

    ## draw losses at the top
    grid_start = 0
    for tech in tech_order:
        df_tmp = df_stats_merged[df_stats_merged["tech"] == tech]
        df_tmp = df_tmp.reindex(order_dict[tech], fill_value=0)
        df_tmp = df_tmp[losses_order]
        df_tmp = df_tmp.loc[order_dict[tech]]

        n_samples_in_tech = len(df_tmp)
        grid_end = grid_start + n_samples_in_tech
        ax = fig.add_subplot(gs[0, grid_start:grid_end])

        df_tmp.plot.bar(
            stacked=True,
            ax=ax,
            width=individual_barplot_width,
            colormap=losses_color_palette,
        )

        # .set_ylim(ylim_dict[variable])
        ax.get_legend().remove()

        # only set title on top row
        ax.set_title(tech_alias_dict[tech], fontsize=18)
        # only set y label on left col
        if tech == tech_order[0]:
            ax.set_ylabel("Fraction of\nReads", fontsize=15)
        else:
            ax.set_ylabel(None)
            ax.set_yticklabels([])

        ax.set(xlabel="")
        ax.set_xticklabels([])

        # start coordinate of next tech is end coordinate of previous tech
        grid_start = grid_end

    handles, labels = plt.gca().get_legend_handles_labels()
    order_index = [5, 4, 3, 2, 1, 0]
    plt.legend(
        [handles[idx] for idx in order_index],
        [labels[idx] for idx in order_index],
        loc=(1.04, 0),
    )

    ## draw variables_list
    grid_start = 0
    for tech in tech_order:
        # print(grid_start)
        # print(tech)
        # subset df to tech
        df_tmp = df_scstats_merged[df_scstats_merged["tech"] == tech]

        # set quick palette
        color = tech_color_palette[tech]
        palette_tmp = {x: color for x in df_scstats_merged["sample_id"].unique()}

        n_samples_in_tech = len(df_tmp["sample_id"].unique())
        # print(n_samples_in_tech)
        grid_end = grid_start + n_samples_in_tech
        for variable in variables_list:
            # print(variable)
            # now determine correct location on gridspec
            ax = fig.add_subplot(
                gs[variables_list.index(variable) + 1, grid_start:grid_end]
            )

            sns.violinplot(
                data=df_tmp,
                x="sample_id",
                y=variable,
                # hue="sample_id",
                order=order_dict[tech],
                palette=palette_tmp,
                ax=ax,
                showfliers=False,
                cut=0,
                bw=0.15,
                inner="box",
                linewidth=1,
            )

            if not variable == "Unique_nr_frag_in_regions_k":
                ax.set_ylim(ylim_dict[variable])
            else:
                ax.set_ylim(ylim_dict[variable])

            # ax.get_legend().remove()

            # only set y label on left col or s3-atac
            if tech == tech_order[0]:
                ax.set_ylabel(var_alias_dict[variable], fontsize=15)
            else:
                ax.set_ylabel(None)
                ax.set_yticklabels([])

            ax.set(xlabel="")
            ax.set_xticklabels([])

            ax.set(xlabel="")
            if variable == variables_list[-1]:
                ax.set_xticklabels(
                    labels=order_dict_tech_ultrashort[tech], rotation=45, ha="right"
                )
            else:
                ax.set_xticklabels([])

        # start coordinate of next tech is end coordinate of previous tech
        grid_start = grid_end

    # plt.rcParams["font.weight"] = "bold"
    plt.tight_layout()
    plt.savefig(png_output_path, dpi=600, facecolor="white", bbox_inches='tight')
    plt.savefig(svg_output_path, dpi=600, facecolor="white", bbox_inches='tight')

    plt.show()
    plt.close()