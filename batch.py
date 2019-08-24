#!/usr/bin/env python3

"""
GROW: Batch Simulator

author: Tighe Costa
email: tighe.costa@gmail.com
"""

import pandas as pd
import argparse

import simulate

def load(configFile, batchSheet):
    """
    Import configuration file, loading cell types spreadsheet and specified
    batch definitions spreadsheet.
    """
    # import excel workbook
    configxlsx = pd.ExcelFile(configFile)

    # pull worksheet with cell type definitions
    cellTypes = pd.read_excel(configxlsx, "cell types", index_col=0)

    # pull worksheet with simulation parameter definitions
    batch = pd.read_excel(configxlsx, batchSheet, index_col=0)

    return cellTypes, batch


def build(cellTypeDefs, sim_params):
    sim = sim_params.to_dict()

    config = sim

    cellTypes = []
    for name in config["cellTypeNames"].split(", "):
        cellTypes.append(cellTypeDefs[name].to_dict())

    config["cellTypes"] = cellTypes

    config["mixRatios"] = [int(x) for x in config["mixRatios"].split(", ")]

    config["outputSize"] = tuple([int(x) for x in config["outputSize"].split(", ")])

    return config


def main(configFile, batchSheet):
    # load configuration file
    cellTypes, batch = load(configFile, batchSheet)

    for sim_name in batch:
        # build configuration dict for simulation
        config = build(cellTypes, batch[sim_name])

        # run simulation
        simulate.main(config)
        print()

    return None


if __name__ == "__main__":
    # set up command line argument parser
    parser = argparse.ArgumentParser(
        usage="%(prog)s [-h] [-c config] -b batch"
    )

    # configure input arguments
    parser.add_argument("-c", "--config", metavar="", type=str,
                        help="path to configuraiton file",
                        default="config.xlsx")
    parser.add_argument("-b", "--batch", metavar="", type=str,
                        help="name of batch worksheet",
                        required=True)

    # process input arguments
    args = parser.parse_args()
    configFile = args.config            # configuraiton file
    batchSheet = args.batch             # batch definitions

    main(configFile, batchSheet)
