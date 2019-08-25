#!/usr/bin/env python3

"""
GROW: Biologically Inspired Cellular Growth Algorithm

author: Tighe Costa
email: tighe.costa@gmail.com
"""

import numpy as np
import random
import sys
from datetime import datetime

import cv2
import imageio
from PIL import Image

import h5py
import pandas as pd

import grow

mapPath = "_food/"
outputPath = "_sims/"

def initialize(config):
    # import environment definitions
    food = cv2.imread(mapPath + config["foodFile"], cv2.IMREAD_GRAYSCALE)
    map = cv2.imread(mapPath + config["mapFile"], cv2.IMREAD_GRAYSCALE)

    # initialize environment
    env = grow.Dish(config["width"], config["height"], food, map,
                    config["cellTypes"])

    # populate environment with seeds
    cells = env.populate(config)

    return env, cells


def draw(frames, fileName, dispSize):
    # scale frames to output size
    outFrames = []
    for frame in frames:
        outFrames.append(cv2.resize(frame, dispSize))

    # write frames to file
    imageio.mimwrite(fileName, outFrames, macro_block_size=8, fps=30, quality=9)

    return None


def save(dataList, labelList, outputPath, config):
    # timestamp file path
    timestamp = datetime.now().strftime("%Y-%m-%d %H_%M_%S ")
    path = outputPath + timestamp + config["batchName"] + "-" + config["name"]

    # initialize file
    outfile = h5py.File(path + " data.h5", "w")

    # save config to file
    # converts each value in key, value pair to string
    cfg = outfile.create_group("config")
    for k, v in config.items():
        cfg.create_dataset(k, data=str(v))

    # save each element in data as dataset with name from label
    for n, data in enumerate(dataList):
        # pull label from list
        label = labelList[n]

        # add dataset to file
        outfile.create_dataset(label, data=np.array(data),
                               compression="gzip", compression_opts=4)

        # render dataset
        draw(data, path + " " + label + ".mp4", config["outputSize"])

    return None


def main(config):
    startTime = datetime.now()
    np.random.seed()

    print("[1/3] Initializing...")
    env, cells = initialize(config)

    print("  Complete.")

    print("[2/3] Growing...")
    framesSpecies = []
    framesFood = []

    t = 0
    nCellsLast = config["seeds"]

    for i in range(1, config["maxIter"]+1):

        chunks = np.array_split(np.array(cells), config["timeWarpFactor"])

        for chunk in chunks:

            # skip empty chunks
            # this happens because numpy will return an empty chunk if there
            # are fewer objects than chunks
            if not chunk.tolist():
                continue

            # grow non-empty chunks
            for n, cell in enumerate(chunk):
                grow.update(cell, env)
                t += 1

            framesSpecies.append(
                (env.species*255/len(config["cellTypes"])).astype("uint8"))
            framesFood.append(
                (np.maximum(env.food*255/100,
                            np.zeros((config["height"], config["width"])))).astype("uint8"))

        # # random shuffle cell update order
        # env.cellList.sort(key=lambda cell: (cell.species, cell.age))
        # random.shuffle(env.cellList)

        # print growth progress to command line
        progress = int(i*76/config["maxIter"])
        sys.stdout.write("\r"+"  ["+"-"*progress+" "*(75-progress)+"]")
        sys.stdout.flush()

        # stop simulation if all cells have died
        if (env.nCells == 0):
            framesSpecies.append(
                (env.species*255/len(config["cellTypes"])).astype("uint8"))
            framesFood.append(
                (np.maximum(env.food*255/100,
                            np.zeros((config["height"], config["width"])))).astype("uint8"))

            print("\n  All cells dead in %i iterations. Terminating simulation." % i)
            break

        # stop simulation if maximum iterations completed
        if i == config["maxIter"]:
            print("\n  Completed %i iterations." % i)

    print("[3/3] Saving...")

    # save out data
    save([framesSpecies, framesFood], ["species", "nutrients"],
         outputPath, config)

    print("  Complete.")

    elapsedTime = str(datetime.now() - startTime)
    config["time"] = elapsedTime
    print("Simulation time: %s" % elapsedTime)

    return None


if __name__ == "__main__":
    print("GROW: Biologically Inspired Cellular Growth Algorithm\n")

    import demo_config
    main(demo_config.config)


# -----------------------------------------------------------------------------
# references:
# Reas, Casey. "Simulate: Diffusion-Limited Aggregation." Form+Code in Design,
#     Art, and Architecture. http://formandcode.com/code-examples/simulate-dla
# -----------------------------------------------------------------------------
