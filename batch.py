#!/usr/bin/env python3

"""
GROW: Batch Simulator

author: Tighe Costa
email: tighe.costa@gmail.com
"""

import pandas as pd
import openpyxl
import argparse

import simulate

def load(configFile, batchName):
    """
    Import configuration file, loading cell types spreadsheet and specified
    batch definitions spreadsheet.
    """
    # import excel workbook
    configxlsx = pd.ExcelFile(configFile)

    # pull worksheet with cell type definitions
    cellTypes = pd.read_excel(configxlsx, "cell types", index_col=0)

    # pull worksheet with simulation parameter definitions
    batch = pd.read_excel(configxlsx, batchName, index_col=0)

    return cellTypes, batch


def build(cellTypeDefs, sim_params, batchName):
    """

    """
    config = sim_params.to_dict()

    cellTypes = []
    for name in config["cellTypeNames"].split(", "):
        cellTypes.append(cellTypeDefs[name].to_dict())

    config["cellTypes"] = cellTypes

    config["mixRatios"] = [int(x) for x in config["mixRatios"].split(", ")]

    config["outputSize"] = tuple([int(x) for x in config["outputSize"].split(", ")])

    # add file name tags
    config["name"] = sim_params.name
    config["batchName"] = batchName

    return config


def save(configFile, batchName, batch):
    # load excel worksheet for writing
    file = openpyxl.load_workbook(configFile)
    sheet = file[batchName]

    for colidx, sim_name in enumerate(batch, start=2):
        sheet.cell(row=2, column=colidx).value = batch[sim_name]["completed"]

    file.save(configFile)

    return None


def main(configFile, batchName):
    # load configuration file
    cellTypes, batch = load(configFile, batchName)

    if all(batch.loc["completed", :].values.tolist()):
        print("Simulation batch " + batchName + " already executed. Terminating program.")
        return None

    for sim_name in batch:
        if batch[sim_name]["completed"] is False:
            print("\nGROW Simulation #" + sim_name)

            # build configuration dict for simulation
            config = build(cellTypes, batch[sim_name], batchName)

            # run simulation
            simulate.main(config)
            batch[sim_name]["completed"] = True

    # save(configFile, batchName, batch)

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
    batchName = args.batch             # batch definitions

    print("GROW: Biologically Inspired Cellular Growth Algorithm")
    main(configFile, batchName)
