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


def initialize(config):
    # import environment definitions
    food = cv2.imread(config["foodFile"], cv2.IMREAD_GRAYSCALE)
    map = cv2.imread(config["mapFile"], cv2.IMREAD_GRAYSCALE)

    # initialize environment
    env = grow.Dish(config["width"], config["height"], food, map,
                    config["cellTypes"])

    # populate environment with seeds
    cells = env.populate(config["seeds"])

    return env, cells


def draw(frames, fileName, dispSize):
    # scale frames to output size
    outFrames = []
    for frame in frames:
        outFrames.append(cv2.resize(frame, dispSize))

    # write frames to file
    imageio.mimwrite(fileName, outFrames, macro_block_size=16, fps=30, quality=9)

    return None


def save(dataList, labelList, outputPath, outputSize):
    # timestamp file path
    timestamp = datetime.now().strftime("%Y-%m-%d %H_%M_%S")
    path = outputPath + timestamp

    # initialize file
    outfile = h5py.File(path + " data.h5", "w")

    # save each element in data as dataset with name from label
    for n, data in enumerate(dataList):
        # pull label from list
        label = labelList[n]

        # add dataset to file
        outfile.create_dataset(label, data=np.array(data),
                               compression="gzip", compression_opts=4)

        # render dataset
        draw(data, path + " " + label + ".mp4", outputSize)

    return None


def main(config):

    np.random.seed()

    print("GROW: Biologically Inspired Cellular Growth Algorithm\n")

    print("[1/3] Initializing...")
    env, cells = initialize(config)

    print("Complete.")

    print("\n[2/3] Growing...")
    framesSpecies = []
    framesFood = []

    t = 0
    nCellsLast = config["seeds"]

    for i in range(1, config["maxIter"]+1):

        chunks = np.array_split(np.array(cells), config["timeWarpFactor"])

        for chunk in chunks:

            if not chunk.tolist():
                continue

            for n, cell in enumerate(chunk):
                grow.update(cell, env)
                t += 1

            framesSpecies.append(
                (env.species*255/len(config["cellTypes"])).astype("uint8"))
            framesFood.append(
                (np.maximum(env.food*255/100,
                            np.zeros((config["height"], config["width"])))).astype("uint8"))

        # random shuffle cells
        # env.cellsList.sort(key=lambda cell: (cell.species, cell.age))
        # random.shuffle(env.cellsList)

        progress = int(i*78/config["maxIter"])
        sys.stdout.write("\r"+"["+"-"*progress+" "*(77-progress)+"]")
        sys.stdout.flush()

        if (env.nCells == 0):
            framesSpecies.append(
                (env.species*255/len(config["cellTypes"])).astype("uint8"))
            framesFood.append(
                (np.maximum(env.food*255/100,
                            np.zeros((config["height"], config["width"])))).astype("uint8"))

            print("\nAll cells dead in %i iterations. Terminating simulation." % i)
            break

        if i == config["maxIter"]:
            print("\nCompleted %i iterations." % i)

    print("\n[3/3] Saving...")

    # save out data
    save([framesSpecies, framesFood], ["species", "nutrients"],
         config["outputPath"], config["outputSize"])

    print("Complete.")

    return


if __name__ == "__main__":
    import demo_config
    main(demo_config.config)


# -----------------------------------------------------------------------------
# references:
# Reas, Casey. "Simulate: Diffusion-Limited Aggregation." Form+Code in Design,
#     Art, and Architecture. http://formandcode.com/code-examples/simulate-dla
# -----------------------------------------------------------------------------
