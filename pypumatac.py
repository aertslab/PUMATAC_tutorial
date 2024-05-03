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
import math

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
    for root, dirs, files in os.walk(startpath, followlinks=True):
        # Exclude hidden directories and files
        dirs[:] = [d for d in dirs if not d[0] == "."]
        files = [f for f in files if not f[0] == "."]

        level = root.replace(startpath, "").count(os.sep)
        if level <= maxlevel:
            indent = " " * 4 * (level)
            print("{}{}/".format(indent, os.path.basename(root)))
        if level == maxlevel:
            dirs[
                :
            ] = (
                []
            )  # clear dir list at max level to prevent unnecessary directory traversal


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
        filename = f"{genome}_annotation.tsv"
        if os.path.exists(filename):
            print(f"Loading cached genome annotation {filename}")
            annotation = pd.read_csv(filename, sep="\t", header=0, index_col=0)
        else:
            dataset = pbm.Dataset(
                name=pbm_genome_name_dict.get(genome, "default_value"),
                host=pbm_host_dict.get(genome, "default_value"),
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
            annotation["Chromosome/scaffold name"] = (
                "chr" + annotation["Chromosome/scaffold name"]
            )
            annotation.columns = [
                "Chromosome",
                "Start",
                "Strand",
                "Gene",
                "Transcript_type",
            ]
            annotation = annotation[annotation.Transcript_type == "protein_coding"]
            annotation.to_csv(filename, sep="\t")

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


def plot_frag_qc(
    x,
    y,
    z,
    ax,
    x_thr_min=None,
    x_thr_max=None,
    y_thr_min=None,
    y_thr_max=None,
    ylab=None,
    xlab="Number of (unique) fragments in regions",
    cmap="viridis",
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

    sp = ax.scatter(
        x,
        y,
        c=z if z is not None else None,
        s=s,
        edgecolors=None,
        marker=marker,
        cmap=cmap,
        **kwargs,
    )
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
    metadata_bc_df,
    bc_passing_filters=[],
    x_thresh=None,
    y_thresh=None,
    include_kde=False,
    detailed_title=True,
    max_dict={},
    min_dict={},
    s=4,
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    y_var_list = ["TSS_enrichment", "FRIP", "Dupl_rate"]
    y_labels = ["TSS Enrichment", "FRIP", "Duplicate rate per cell"]

    # plot everything
    axes = [ax1, ax2, ax3]
    for i, (y_var, ax, y_label) in enumerate(zip(y_var_list, axes, y_labels)):
        z_col_name = f"kde__log_Unique_nr_frag_in_regions__{y_var}"
        # print(metadata_bc_df.columns)
        if include_kde:
            print("plotting with KDE")
            if not z_col_name in metadata_bc_df.columns:
                print(f"{z_col_name} is not present, calculating")
                x_log = np.log(metadata_bc_df["Unique_nr_frag_in_regions"] + 1)
                xy = np.vstack([x_log, metadata_bc_df[y_var]])
                # print(xy)
                z = gaussian_kde(xy)(xy)
                # print(z)

                # now order x and y in the same way that z was ordered, otherwise random z value is assigned to barcode:
                idx = (
                    z.argsort()
                )  # order based on z value so that highest value is plotted on top, and not hidden by lower values
                df_sub = pd.DataFrame(index=metadata_bc_df.index[idx])
                df_sub[z_col_name] = z[idx]
                metadata_bc_df[z_col_name] = df_sub[z_col_name]

            else:
                print(f"{z_col_name} is present, not calculating")
        else:
            print("plotting without KDE")

        if include_kde:
            metadata_bc_df = metadata_bc_df.sort_values(by=z_col_name, ascending=True)

        plot_frag_qc(
            x=metadata_bc_df["Unique_nr_frag_in_regions"],
            y=metadata_bc_df[y_var],
            z=metadata_bc_df[z_col_name] if include_kde else None,
            ylab=y_label,
            s=s,
            x_thr_min=x_thresh,
            y_thr_min=y_thresh,
            xlim=[10, max_dict["Unique_nr_frag_in_regions"]],
            ylim=[0, max_dict[y_var]] if y_var == "TSS_enrichment" else [0, 1],
            ax=ax,
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
        title = f"{sample_alias}: Kept {len(bc_passing_filters)} cells using Otsu filtering. Median Unique Fragments: {med_nf:.0f}. Median TSS Enrichment: {med_tss:.2f}. Median FRIP: {med_frip:.2f}\nUsed a minimum of {x_thresh:.2f} fragments and TSS enrichment of {y_thresh:.2f})"
    else:
        title = sample

    fig.suptitle(title, x=0.5, y=0.95, fontsize=10)
    return fig


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

    # Read cell barcode (column 4) and counts (column 5) per fragment from fragments BED file.
    fragments_df = pl.read_csv(
        fragments_bed_filename,
        has_header=False,
        skip_rows=skip_rows,
        separator="\t",
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
    n_reads,
    min_uniq_frag=200,
    selected_barcodes=[],
    sampling_fractions=sampling_fractions_default,
    stats_tsv_filename="sampling_stats.tsv",
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
            fragments_all_df.sample(fraction=sampling_fraction),
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

    # then add some extra stats
    stats_df["total_reads"] = n_reads * stats_df.index

    stats_df["mean_reads_per_barcode"] = (
        stats_df["total_reads"] / stats_df["cell_barcode_count"]
    )
    stats_df["mean_reads_per_barcode"].fillna(0, inplace=True)
    stats_df["duplication_rate"] = (
        stats_df["total_frag_count"] - stats_df["total_unique_frag_count"]
    ) / stats_df["total_frag_count"]
    stats_df["duplication_rate"] = stats_df["duplication_rate"].fillna(0)

    print(f'Saving statistics in "{stats_tsv_filename}".')
    stats_df.to_csv(stats_tsv_filename, sep="\t")

    return stats_df


# subsample a fragments file and return the subsampled fragments file
def sub_sample_fragments_single(
    fragments_df,
    sampling_fraction=None,
    out_file_path=None,
):
    if sampling_fraction > 1:
        print("sampling fraction > 1, impossible, returning none")

    elif sampling_fraction == 1:
        print("sampling fraction = 1, returning full fragments file.")

        if out_file_path:
            fragments_df.write_csv(out_file_path, separator="\t", has_header=False)
        else:
            return fragments_df

    elif sampling_fraction == 0.0:
        print("sampling fraction = 0, returning none")
        return None

    else:
        fragments_all_df = fragments_df.with_columns(
            pl.col("FragmentCount").repeat_by(pl.col("FragmentCount"))
        ).explode("FragmentCount")

        # downsample
        fragments_sampled_df = fragments_all_df.sample(fraction=sampling_fraction)

        # re-group
        fragments_sampled_df_contracted = fragments_sampled_df.groupby(
            ["Chromosome", "Start", "End", "CellBarcode"]
        ).agg([pl.count("FragmentCount").alias("FragmentCount")])

        if out_file_path:
            fragments_sampled_df_contracted.write_csv(
                out_file_path, separator="\t", has_header=False
            )

        else:
            return fragments_sampled_df_contracted


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
    if x_axis == "mean_reads_per_barcode":
        x_data = np.array(stats_df.loc[0:, x_axis]) / 10**3
    else:
        x_data = np.array(stats_df.loc[0:, x_axis])

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
        x_fit, function(x_fit, *best_fit_ab), label="fitted", c="black", linewidth=1
    )
    # plot raw data
    ax.scatter(x=x_data, y=y_data, c="red", s=10)

    # mark curent saturation
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
    upper_xticklabels = [str(int(x)) for x in ax.get_xticks() * n_cells / 1000]
    ax2.set_xticklabels(upper_xticklabels)
    ax2.set_xlabel("Total reads (millions)")
    ax2.set_xlim([0, x_max])

    # save figure
    ax.set_xlabel("Reads per cell (thousands)")

    # plt.yscale("log")
    # ax.set_xscale("log")

    ax.set_ylabel(y_axis)
    title_str = f"{sample}\n{n_cells} cells, {round(n_reads/1000000)}M reads\nCurrently at {int(curr_y_coef)} {y_axis} with {int(curr_x_coef)} kRPC\nFor {int(percentage_toplot*100)}% saturation: {int(x_coef*n_cells/1000)}M reads needed\n"
    ax.set_title(title_str)

    plt.savefig(png_output_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_output_path, dpi=300, bbox_inches="tight")
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

    if x_axis == "mean_reads_per_barcode":
        x_data = np.array(stats_df.loc[0:, x_axis]) / 10**3
    else:
        x_data = np.array(stats_df.loc[0:, x_axis])
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
    upper_xticklabels = [str(int(x)) for x in ax.get_xticks() * n_cells / 1000]
    ax2.set_xticklabels(upper_xticklabels)
    ax2.set_xlabel("Total reads (millions)")
    ax2.set_xlim([0, x_max])

    ax.set_xlabel("Reads per cell (thousands)")
    # plt.yscale("log")
    # ax.set_xscale("log")

    ax.set_ylabel("Duplication rate (%)")
    title_str = f"{sample}\n{n_cells} cells, {round(n_reads/1000000)}M reads\nCurrently at {int(curr_y_coef*100)}% duplication rate with {int(curr_x_coef)} kRPC\nFor {int(percentage_toplot*100)}% duplication rate, {int(x_coef*n_cells/1000)}M reads needed\n"
    ax.set_title(title_str)

    plt.savefig(png_output_path, dpi=300, bbox_inches="tight")
    plt.savefig(svg_output_path, dpi=300, bbox_inches="tight")
    plt.show()

    plt.close()
    print(title_str)


### Parsing data
def load_file(file_path, delimiter):
    if os.path.exists(file_path):
        return pd.read_csv(file_path, sep=delimiter, engine="python", index_col=0)
    else:
        print(f"{file_path} does not exist!")
        return None


def print_verbose_info(data):
    print("-------------------------------------\n")
    for key, value in data.items():
        print(f"{key}: {value}")
    print("-------------------------------------\n")


def get_correct_barcodes(df, verbose):
    nreads, percentage_correct_barcodes = calculate_barcode_metrics(df)

    if verbose:
        print_verbose_info(
            {
                "nreads": nreads,
                "nbarcodes_total": nbarcodes_total,
                "percentage_correct_barcodes": percentage_correct_barcodes,
            }
        )

    return nreads, percentage_correct_barcodes


def get_mapping_stats(df, verbose):
    percent_mapq30 = (
        df.loc["Reads mapped with MAPQ>30:"] / df.loc["raw total sequences:"] * 100
    )
    avg_insert = df.loc["insert size average:"]
    avg_map_quality = df.loc["average quality:"]
    r1_length = df.loc["maximum first fragment length:"]
    r2_length = df.loc["maximum last fragment length:"]

    if verbose:
        print_verbose_info(
            {
                "read 1 length": int(r1_length),
                "read 2 length": int(r2_length),
                "average map quality": round(avg_map_quality, 2),
                "percent mapq30": round(percent_mapq30, 2),
                "insert size average": avg_insert,
            }
        )

    return r1_length, r2_length, avg_insert, percent_mapq30, avg_map_quality


def get_pipeline_stats(stats_dict, verbose):
    for sample, filepath in stats_dict.items():
        df = load_file(filepath, ",")
        if df is not None:
            percentage_correct_barcodes = df["ATAC Valid barcodes"][0] * 100
            n_reads = df["ATAC Sequenced read pairs"][0]
            percent_mapq30 = df["ATAC Confidently mapped read pairs"][0] * 100

            if verbose:
                print_verbose_info(
                    {
                        "percentage_correct_barcodes": percentage_correct_barcodes,
                        "percent mapq30": round(percent_mapq30, 2),
                    }
                )

            yield sample, percentage_correct_barcodes, n_reads, percent_mapq30


def collect_and_sum_barcode_stats(files):
    print(f"loading barcode stats files: {files}")
    df_total = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, sep="\t\t|\t", engine="python", index_col=0, header=None)

        if df_total.empty:
            df_total = df.copy()
        else:
            df_total = pd.concat([df_total, df], axis=1)

    df_total["total"] = df_total.sum(axis=1)
    return df_total


def collect_and_sum_mapping_stats(files):
    print(f"loading mapping stats files: {files}")
    df_total = pd.DataFrame()
    for file in files:
        df = pd.read_csv(file, sep="\t\t|\t", engine="python", index_col=0, header=0)

        if df_total.empty:
            df_total = df.copy()
        else:
            df_total = pd.concat([df_total, df], axis=1)

    df_total["total"] = df_total.sum(axis=1)
    return df_total


def calculate_weighted_avg(file_list, value_col, weight_col):
    """
    Function to calculate the weighted average
    """
    df_list = []
    for file in file_list:
        df = load_file(file, "\t")
        df_list.append(df)

    # concatenate all dfs
    df_all = pd.concat(df_list, axis=0)
    # Convert values to float for computation
    df_all[value_col] = df_all[value_col].astype(float)
    df_all[weight_col] = df_all[weight_col].astype(float)
    # calculate weighted average
    return (df_all[value_col] * df_all[weight_col]).sum() / df_all[weight_col].sum()


def get_weighted_averages(sample_name, dir="."):
    # Find all files that contain the sample_name in the file name
    files = glob.glob(os.path.join(dir, f"*{sample_name}*.tsv"))
    # Value columns to be averaged
    value_cols = [
        "r1_length",
        "r2_length",
        "avg_insert_size",
        "%_mapq30",
        "avg_map_quality",
    ]
    weighted_avgs = {}
    for col in value_cols:
        weighted_avgs[col] = calculate_weighted_avg(files, col, "raw total sequences")
    return weighted_avgs


def scrape_mapping_stats(pumatac_output_dir,cr_output_dir, selected_barcodes_path_dict, pipeline_dict, verbose):
    df_stats = pd.DataFrame(index=pd.Index(pipeline_dict.keys()))
    for sample, pipeline in pipeline_dict.items():
        try:
            df = load_file(selected_barcodes_path_dict[sample], "\t")
            df_stats.loc[sample, "sample_id"] = sample
            df_stats.loc[sample, "n_cells"] = len(df)
        except:
            print(
                f"{sample} bc_passing_filters_otsu.txt not found!"
            )

        if pipeline == "PUMATAC":
            # bc stats
            files = glob.glob(
                f"{pumatac_output_dir}/data/reports/barcode/{sample}*.corrected.bc_stats.log"
            )
            df_total = collect_and_sum_barcode_stats(files)
            if df is not None:
                nreads = df_total.at["nbr_reads:", "total"]
                try:
                    nbarcodes_total = df_total.at[
                        "nbr_reads_with_bc1_bc2_bc3_correct_or_correctable", "total"
                    ]
                except:
                    nbarcodes_total = df_total.at["total_bc_found", "total"]

                percentage_correct_barcodes = nbarcodes_total / nreads * 100

                df_stats.loc[sample, "n_reads"] = nreads
                df_stats.loc[sample, "%_correct_barcodes"] = round(
                    percentage_correct_barcodes, 2
                )

            # mapping stats
            files = glob.glob(
                f"{pumatac_output_dir}/data/reports/mapping_stats/{sample}*.mapping_stats.tsv"
            )
            df_total = collect_and_sum_mapping_stats(files)
            if df_total is not None:
                percent_mapq30 = (
                    df_total.at["Reads mapped with MAPQ>30:", "total"]
                    / df_total.at["raw total sequences:", "total"]
                    * 100
                )
                avg_insert = df_total.at["insert size average:", "total"]
                avg_map_quality = df_total.at["average quality:", "total"]
                r1_length = df_total.at["maximum first fragment length:", "total"]
                r2_length = df_total.at["maximum last fragment length:", "total"]

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
                df_stats.loc[sample, "%_mapq30"] = round(percent_mapq30, 2)
                df_stats.loc[sample, "avg_map_quality"] = round(avg_map_quality, 2)

        elif pipeline == "cellranger-atac":
            filepath = f"{cr_output_dir}/{sample}/outs/summary.csv"
            if os.path.exists(filepath):
                df = pd.read_csv(filepath)
                percentage_correct_barcodes = df["Valid barcodes"][0] * 100
                n_reads = df["Sequenced read pairs"][0]
                percent_mapq30 = df["Confidently mapped read pairs"][0] * 100

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
                print(f"{filepath} does not exist!")

        elif pipeline == "cellranger-arc":
            filepath = f"{cr_output_dir}/{sample}/outs/summary.csv"

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
                print(f"{filepath} does not exist!")

    return df_stats

import pandas as pd
import pickle


def scrape_scstats(metadata_path_dict, selected_cells_path_dict, df_stats):
    cols_to_ignore = ['kde__log_Unique_nr_frag_in_regions__TSS_enrichment', 'kde__log_Unique_nr_frag_in_regions__FRIP', 'kde__log_Unique_nr_frag_in_regions__Dupl_rate']

    df_merged = pd.DataFrame()
    df_scstats_merged = pd.DataFrame()

    for sample in metadata_path_dict.keys():
        with open(metadata_path_dict[sample], "rb") as f:
            df = pickle.load(f)

        with open(selected_cells_path_dict[sample], "rb") as f:
            selected_barcodes = pickle.load(f)

        df = df.loc[selected_barcodes]

        # Create a DataFrame from df_median
        df_median = df.median(numeric_only=True).to_frame().T

        # Add new fields to df_median
        df_median.columns = ["Median_" + x.lower() for x in df_median.columns]
        try:
            frac_barcodes_merged = len([x for x in df.index if "_" in x.split("__")[0]])/ len(df)
        except ZeroDivisionError:
            frac_barcodes_merged = 0
            
        df_median = df_median.assign(
            total_nr_frag_in_selected_barcodes=df["Total_nr_frag"].sum(),
            total_nr_unique_frag_in_selected_barcodes=df["Unique_nr_frag"].sum(),
            total_nr_unique_frag_in_selected_barcodes_in_regions=df[
                "Unique_nr_frag_in_regions"
            ].sum(),
            n_barcodes_merged=len([x for x in df.index if "_" in x.split("__")[0]]),
            frac_barcodes_merged=frac_barcodes_merged,
        )

        df_median["sample"] = sample  # Add sample name to df_median
        df_merged = pd.concat([df_merged, df_median])
        df_scstats_merged = pd.concat([df_scstats_merged, df])

    df_merged.set_index("sample", inplace=True)  # Set 'sample' as index

    # Select subset of columns
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

    df_scstats_merged_benchmark = df_scstats_merged_benchmark[[x for x in df_scstats_merged.columns if x not in cols_to_ignore]]

    df_scstats_merged = pd.concat([df_scstats_merged_benchmark, df_scstats_merged])

    df_scstats_merged["Unique_nr_frag_in_regions_k"] = (
        df_scstats_merged["Unique_nr_frag_in_regions"] / 1000
    )

    df_stats = pd.concat([df_stats, df_merged], axis=1)
    return df_scstats_merged, df_stats


### Calculate losses


def calculate_losses(df_stats, df_scstats_merged):
    grouped_scstats = df_scstats_merged.groupby("sample_id")
    total_nr_frag = grouped_scstats["Total_nr_frag"].sum()
    unique_nr_frag = grouped_scstats["Unique_nr_frag"].sum()
    unique_nr_frag_regions = grouped_scstats["Unique_nr_frag_in_regions"].sum()

    df_stats["total_nr_frag_in_selected_barcodes"] = df_stats.index.map(total_nr_frag)
    df_stats["total_nr_unique_frag_in_selected_barcodes"] = df_stats.index.map(
        unique_nr_frag
    )
    df_stats[
        "total_nr_unique_frag_in_selected_barcodes_in_regions"
    ] = df_stats.index.map(unique_nr_frag_regions)

    df_stats["with_correct_barcode"] = (
        df_stats["n_reads"] * df_stats["%_correct_barcodes"] / 100
    )
    df_stats["mapped"] = df_stats["with_correct_barcode"] * df_stats["%_mapq30"] / 100

    numeric_cols = [
        "total_nr_frag_in_selected_barcodes",
        "total_nr_unique_frag_in_selected_barcodes",
        "total_nr_unique_frag_in_selected_barcodes_in_regions",
        "with_correct_barcode",
        "mapped",
    ]

    df_stats[numeric_cols] = df_stats[numeric_cols].div(df_stats["n_reads"], axis=0)

    df_stats["No correct barcode"] = 1 - df_stats["with_correct_barcode"]
    df_stats["Not mapped properly"] = (
        df_stats["with_correct_barcode"] - df_stats["mapped"]
    )
    df_stats["Fragments in background noise barcodes"] = (
        df_stats["mapped"] - df_stats["total_nr_frag_in_selected_barcodes"]
    )
    df_stats["Duplicate fragments in cells"] = (
        df_stats["total_nr_frag_in_selected_barcodes"]
        - df_stats["total_nr_unique_frag_in_selected_barcodes"]
    )
    df_stats["Unique fragments in cells, not in peaks"] = (
        df_stats["total_nr_unique_frag_in_selected_barcodes"]
        - df_stats["total_nr_unique_frag_in_selected_barcodes_in_regions"]
    )
    df_stats["Unique fragments in cells and in peaks"] = df_stats[
        "total_nr_unique_frag_in_selected_barcodes_in_regions"
    ]

    if not "No correct barcode" in df_stats.columns:
        df_stats = pd.concat([df_stats, df_sub], axis=1)

    df_stats_reference = pd.read_csv(
        "PUMATAC_dependencies/data/fixedcells_general_statistics.tsv",
        sep="\t",
        index_col=0,
    )

    df_stats_reference = df_stats_reference.assign(
        tech=df_stats_reference["technology"],
        sample_id=df_stats_reference.index,
        n_cells=df_stats_reference["cells"],
        n_reads=df_stats_reference["reads"],
    )

    df_stats_reference["with_correct_barcode"] = (
        df_stats_reference["n_reads"] * df_stats_reference["%_correct_barcodes"] / 100
    )
    df_stats_reference["mapped"] = (
        df_stats_reference["with_correct_barcode"]
        * df_stats_reference["%_mapq30"]
        / 100
    )
    df_stats_reference = df_stats_reference[df_stats.columns]
    df_stats_reference.index = df_stats_reference.index + ".FIXEDCELLS"
    df_stats_reference["tech"] = df_stats_reference.index.str.split("_").str[1]

    df_stats["tech"] = "user_sample"
    df_stats_merged = pd.concat([df_stats_reference, df_stats])

    df_stats_merged.rename(
        columns={
            "Fragments in background noise barcodes": "Fragments in noise barcodes",
            "Unique fragments in cells, not in peaks": "Unique, in cells, not in peaks",
            "Unique fragments in cells and in peaks": "Unique, in cells, in peaks",
        },
        inplace=True,
    )

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
    n_samples = len(df_stats_merged.index)
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
    plt.savefig(png_output_path, dpi=600, facecolor="white", bbox_inches="tight")
    plt.savefig(svg_output_path, dpi=600, facecolor="white", bbox_inches="tight")

    plt.show()
    plt.close()


def find_nearest(array, value):
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return array[idx]


def get_fit(
    filepath,
    sample,
    n_reads,
    n_cells,
    to_downsample,
    x_axis="mean_reads_per_barcode",
    y_axis="median_uniq_frag_per_bc",
    function=MM,
    maxfev=5000,
):
    stats_df = pd.read_csv(filepath, sep="\t", index_col=0)

    if x_axis == "mean_reads_per_barcode":
        x_data = np.array(stats_df.loc[0:, x_axis]) / 10**3
    else:
        x_data = np.array(stats_df.loc[0:, x_axis])

    y_data = np.array(stats_df.loc[0:, y_axis])
    # fit to MM function

    best_fit_ab, covar = curve_fit(
        function, x_data, y_data, bounds=(0, +np.inf), maxfev=maxfev
    )

    # expand fit space
    x_fit = np.linspace(0, int(np.max(x_data) * 1000), num=100000)
    y_fit = function(x_fit, *(best_fit_ab))

    # Find read count for percent given depth
    curr_x_idx = list(x_fit).index(find_nearest(x_fit, to_downsample / 1000))
    curr_y_coef = y_fit[curr_x_idx]

    return curr_y_coef


def plot_losses_downsampled(
    df_stats_merged,
    sample_order,
    sample_alias_dict,
    tech_order,
    svg_output_path,
    png_output_path,
    individual_barplot_width=0.5,
    individual_plot_row_height=4,
    dpi=300,
    depth=None,
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
        "user_sample": [sample_alias_dict[x] for x in sample_order],
    }

    losses_order_downsampled = [
        "tech",
        "No correct barcode",
        "Not mapped properly",
        "Fragments in noise barcodes",
        "Duplicate fragments in cells (downsampled)",
        "Unique, in cells, not in peaks (downsampled)",
        "Unique, in cells, in peaks (downsampled)",
    ]

    losses_order_downsampled = losses_order_downsampled[::-1]

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
    n_var = 1

    fig = plt.figure(
        figsize=(
            individual_barplot_width * (n_samples),
            individual_plot_row_height * n_var,
        ),
        dpi=dpi,
    )

    gs = GridSpec(
        1,
        n_samples,
        figure=fig,
    )

    ## draw losses at the top
    grid_start = 0
    for tech in tech_order:
        df_tmp = df_stats_merged[df_stats_merged["tech"] == tech]
        df_tmp.index = [sample_alias_dict[x] for x in df_tmp.index]
        df_tmp = df_tmp[losses_order_downsampled]
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
        ax.set_title(f"User samples at {int(depth/1000)}k RPC", fontsize=18)
        # only set y label on left col
        if tech == tech_order[0]:
            ax.set_ylabel("Fraction of\nReads", fontsize=15)
        else:
            ax.set_ylabel(None)
            ax.set_yticklabels([])

        ax.set(xlabel="")
        plt.xticks(rotation=45, ha="right")

        # start coordinate of next tech is end coordinate of previous tech
        grid_start = grid_end

    handles, labels = plt.gca().get_legend_handles_labels()
    order_index = [5, 4, 3, 2, 1, 0]
    plt.legend(
        [handles[idx] for idx in order_index],
        [labels[idx] for idx in order_index],
        loc=(1.04, 0),
    )

    # plt.rcParams["font.weight"] = "bold"
    plt.tight_layout()
    plt.savefig(png_output_path, dpi=dpi, facecolor="white", bbox_inches="tight")
    plt.savefig(svg_output_path, dpi=dpi, facecolor="white", bbox_inches="tight")

    plt.show()
    plt.close()


def qc_mega_plot(
    metadata_bc_pkl_path_dict={},
    sample_order=[],
    n_cells_dict = None,
    include_kde=False,
    x_var=None,
    y_var=None,
    x_label=None,
    y_label=None,
    x_threshold_dict={},
    y_threshold_dict={},
    min_dict={},
    max_dict={},
    alias_dict={},
    n_cols=6,
    figheight=22,
    figwidth=18,
):
    n_rows = math.ceil(len(sample_order) / n_cols)
    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(figwidth, figheight)
    )  # , sharex=True, sharey=True)
    axes = axes.flatten()
    n_samples = len(metadata_bc_pkl_path_dict)
    
    for i in range(n_samples, n_rows * n_cols):
        fig.delaxes(axes[i])
        
    z_col_name = f"kde__log_{x_var}__{y_var}"

    for sample in sample_order:
        ax = axes[sample_order.index(sample)]
        print(f"\tLoading {metadata_bc_pkl_path_dict[sample]}")
        with open(metadata_bc_pkl_path_dict[sample], "rb") as fh:
            metadata_bc_df = pickle.load(fh)
        
        if include_kde:
            if not z_col_name in metadata_bc_df.columns:
                print(f"{z_col_name} is not present, calculating")
                x_log = np.log(metadata_bc_df[x_var] + 1)
                xy = np.vstack([x_log, metadata_bc_df[y_var]])
                # print(xy)
                z = gaussian_kde(xy)(xy)
                # print(z)

                # now order x and y in the same way that z was ordered, otherwise random z value is assigned to barcode:
                idx = (
                    z.argsort()
                )  # order based on z value so that highest value is plotted on top, and not hidden by lower values
                df_sub = pd.DataFrame(index=metadata_bc_df.index[idx])
                df_sub[z_col_name] = z[idx]
                metadata_bc_df[z_col_name] = df_sub[z_col_name]
                
            metadata_bc_df = metadata_bc_df.sort_values(by=z_col_name, ascending=True)

        plot_frag_qc(
            x=metadata_bc_df[x_var],
            y=metadata_bc_df[y_var],
            z=metadata_bc_df[z_col_name] if include_kde else None,
            xlab=x_label,
            ylab=y_label,
            s=3,
            x_thr_min=x_threshold_dict[sample]
            if x_threshold_dict[sample] is not None
            else None,
            y_thr_min=y_threshold_dict[sample]
            if y_threshold_dict[sample] is not None
            else None,
            xlim=[10, max_dict[x_var]],
            ylim=[0, max_dict[y_var]] if y_var == "TSS_enrichment" else [0, 1],
            ax=ax,
        )
        n_cells = n_cells_dict[sample]
        
        ax.set_title(f"{alias_dict[sample]}\n{n_cells} cells")
        sns.despine(ax=ax, top=True, right=True)
    plt.tight_layout()