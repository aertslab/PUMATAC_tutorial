#!/usr/bin/env python3
# written by gert hulselmans

import argparse
import json
import os.path
import sys

from collections import OrderedDict
from operator import itemgetter


def extract_demultiplex_statistics(demultiplex_statistics_json_filename, demultiplex_statistics_tsv_filename):
    # Read demultiplex statistics JSON file.
    with open(demultiplex_statistics_json_filename, 'r') as json_fh, \
            open(demultiplex_statistics_tsv_filename, 'w') as tsv_fh:
        stats = json.load(json_fh)

        # Store number of reads per sample over all lanes.
        nbr_of_reads_per_sample_dict = OrderedDict()

        print('Demuliplex statistics', end='\n', file=tsv_fh)

        # Print number of reads per sample for each lane.
        for conversion_results_per_lane in stats['ConversionResults']:
            # Print lane header.
            print('\n\nLane {0:d}:'.format(conversion_results_per_lane['LaneNumber']),
                  end='\n\n',
                  file=tsv_fh)

            for conversion_results_per_sample_per_lane in conversion_results_per_lane['DemuxResults']:
                # Print sample name and number of reads for the current lane.
                print(conversion_results_per_sample_per_lane['SampleName'],
                      conversion_results_per_sample_per_lane['NumberReads'],
                      sep='\t',
                      end='\n',
                      file=tsv_fh)

                # Add number of reads per sample to have the total number of reads for a sample over all lanes.
                if conversion_results_per_sample_per_lane['SampleName'] in nbr_of_reads_per_sample_dict:
                    nbr_of_reads_per_sample_dict[conversion_results_per_sample_per_lane['SampleName']] \
                        += conversion_results_per_sample_per_lane['NumberReads']
                else:
                    nbr_of_reads_per_sample_dict[conversion_results_per_sample_per_lane['SampleName']] \
                        = conversion_results_per_sample_per_lane['NumberReads']

            # Print number of undetermined reads for the current lane.
            print('Undetermined',
                  conversion_results_per_lane['Undetermined']['NumberReads'],
                  sep='\t',
                  end='\n',
                  file=tsv_fh)

            # Add number of undetermined reads to have the total number of undetermined reads over all lanes.
            if 'Undetermined' in nbr_of_reads_per_sample_dict:
                nbr_of_reads_per_sample_dict['Undetermined'] \
                    += conversion_results_per_lane['Undetermined']['NumberReads']
            else:
                nbr_of_reads_per_sample_dict['Undetermined'] \
                    = conversion_results_per_lane['Undetermined']['NumberReads']

        # Print all lanes header.
        print('\n\nAll lanes:', end='\n\n', file=tsv_fh)

        for sample_name, nbr_reads in nbr_of_reads_per_sample_dict.items():
            # Print sample name and number of reads over all lanes.
            print(sample_name, nbr_reads, sep='\t', end='\n', file=tsv_fh)

        all_unknown_barcodes_dict = {}

        # Print unknown barcodes header.
        print('\n\nUnknown barcodes:', end='\n\n', file=tsv_fh)

        # Loop over all unknown barcodes found in each lane.
        for unknown_barcodes_from_lane in stats['UnknownBarcodes']:
            for barcode, amount in unknown_barcodes_from_lane['Barcodes'].items():
                if barcode in all_unknown_barcodes_dict:
                    all_unknown_barcodes_dict[barcode] += amount
                else:
                    all_unknown_barcodes_dict[barcode] = amount

        for barcode, amount in sorted(all_unknown_barcodes_dict.items(),
                                      key=itemgetter(1),
                                      reverse=True):
            # Print unknown barcode and amount.
            print(barcode, amount, sep='\t', end='\n', file=tsv_fh)


def main():
    parser = argparse.ArgumentParser(
        description='Extract demultiplex statistics from Stats/Stats.json created by bcl2fastq.'
    )
    parser.add_argument(
        '-i',
        '--bcl2fastq-output-dir',
        dest='bcl2fastq_output_dir',
        action='store',
        type=str,
        required=True,
        help='Output directory to which bcl2fastq wrote the demultiplexed FASTQ files. '
             'This directory should contain "Stats/Stats.json".'
    )
    parser.add_argument(
        '-o',
        '--demultiplex-stats',
        dest='demultiplex_statistics_tsv_filename',
        action='store',
        type=str,
        required=True,
        help='Demultiplex statistics TSV output filename.'
    )
    args = parser.parse_args()

    demultiplex_statistics_json_filename = os.path.join(args.bcl2fastq_output_dir, 'Stats', 'Stats.json')

    if not os.path.exists(demultiplex_statistics_json_filename):
        print(
            'Error: Demultiplex statistics JSON input file "{0:s}" could not be found.'.format(
                demultiplex_statistics_json_filename),
            file=sys.stderr
        )
        sys.exit(1)

    if os.path.exists(args.demultiplex_statistics_tsv_filename):
        print(
            'Error: Demultiplex statistics TSV output file "{0:s}" already exists.'.format(
                args.demultiplex_statistics_tsv_filename),
            file=sys.stderr
        )
        sys.exit(1)

    extract_demultiplex_statistics(
        demultiplex_statistics_json_filename=demultiplex_statistics_json_filename,
        demultiplex_statistics_tsv_filename=args.demultiplex_statistics_tsv_filename
    )


if __name__ == "__main__":
    main()
