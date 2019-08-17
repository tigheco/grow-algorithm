#!/usr/bin/env python3

"""
GROW: Biologically Inspired Cellular Growth Algorithm

author: Tighe Costa
email: tighe.costa@gmail.com
"""

import numpy as np
import random
import sys

import cv2
import imageio
from PIL import Image
from datetime import datetime
import h5py

import grow


def initialize(width, height, cellTypes, seeds, foodFile, mapFile, mixRatios):
    # initialize environment
    food = cv2.imread(foodFile, cv2.IMREAD_GRAYSCALE)
    map = cv2.imread(mapFile, cv2.IMREAD_GRAYSCALE)
    env = grow.Dish(width, height, food, map, cellTypes)

    # initialize tissues
    cells = env.addTissues(seeds)

    return env, cells


def draw(frames, fileName, fieldSize, dispSize):
    outFrames = []
    for frame in frames:
        outFrames.append(cv2.resize(frame, dispSize))

    imageio.mimwrite(fileName, outFrames, macro_block_size=16, fps=30, quality=9)

    return None


def main():

    np.random.seed()

    print("GROW: Biologically Inspired Cellular Growth Algorithm\n")

    # -------------------------------------------------------------------------
    # user controls
    width = 112                                 # environment width
    height = 112                                # environment height
    maxIter = 20                               # timeout iterations
    seeds = int(width/10)                       # number of seed cells
    foodFile = "../_food/foodMaps-00.png"       # food map file path
    mapFile =  "../_food/foodMaps-00.png"       # area map file path
    mixRatios = [5, 4, 7]                       # species probability ratios
    cellTypes = [                               # species properties
        {
         "species": 1,
         "proliferation rate": 1,
         "metabolism": 10,
         "abundance": 1,
         "food to divide": 10*4,
         "food to move": 10,
         "division recovery time": 10,
         "food to survive": 10*2,
         "endurance": 180,
        },
        {
         "species": 2,
         "proliferation rate": 1,
         "metabolism": 15,
         "abundance": 1,
         "food to move": 15,
         "food to divide": 15*4,
         "division recovery time": 10,
         "food to survive": 15*2,
         "endurance": 160,
        },
        {
         "species": 3,
         "proliferation rate": 1,
         "metabolism": 20,
         "abundance": 1,
         "food to move": 20,
         "food to divide": 20*4,
         "division recovery time": 10,
         "food to survive": 20*2,
         "endurance": 200,
        }
    ]
    outputSize = 112, 112
    # -------------------------------------------------------------------------

    print("[1/3] Initializing...")
    env, cells = initialize(width, height, cellTypes, seeds, foodFile,
                              mapFile, mixRatios)
    print("Complete.")

    print("\n[2/3] Growing...")
    framesSpecies = []
    framesFood = []

    t = 0
    nCellsLast = seeds

    for i in range(1, maxIter+1):

        chunks = np.array_split(np.array(cells), 2)

        for chunk in chunks:

            if not chunk.tolist():
                continue

            for n, cell in enumerate(chunk):
                grow.update(cell, env)
                t += 1

            framesSpecies.append(
                (env.species*255/len(cellTypes)).astype("uint8"))
            framesFood.append(
                (np.maximum(env.food*255/100,
                            np.zeros((height, width)))).astype("uint8"))

        # random shuffle cells
        # env.cellsList.sort(key=lambda cell: (cell.species, cell.age))
        # random.shuffle(env.cellsList)

        progress = int(i*78/(maxIter-1))
        sys.stdout.write("\r"+"["+"-"*progress+" "*(77-progress)+"]")
        sys.stdout.flush()

        if (env.nCells == 0):
            framesSpecies.append(
                (env.species*255/len(cellTypes)).astype("uint8"))
            framesFood.append(
                (np.maximum(env.food*255/100,
                            np.zeros((height, width)))).astype("uint8"))

            print("\nAll cells dead in %i iterations. Terminating simulation." % i)
            break

        if i == maxIter:
            print("\nCompleted %i iterations." % i)

    print("\n[3/3] Saving...")

    # unique time stamps for data
    outpath = "sims/"
    timestamp = datetime.now().strftime("%Y-%m-%d %H-%M-%S")

    # save out data
    outfile = h5py.File(outpath + timestamp + "data.h5", "w")
    outfile.create_dataset("species",
                           data=np.array(framesSpecies),
                           compression="gzip", compression_opts=4)
    outfile.create_dataset("nutrients",
                           data=np.array(framesFood),
                           compression="gzip", compression_opts=4)

    # create movies
    draw(framesSpecies, outpath + timestamp + " species.mp4", (width, height), outputSize)
    draw(framesFood, outpath + timestamp + " nutrients.mp4", (width, height), outputSize)

    print("Complete.")

    return


if __name__ == "__main__":
    main()


# -----------------------------------------------------------------------------
# references:
# Reas, Casey. "Simulate: Diffusion-Limited Aggregation." Form+Code in Design,
#     Art, and Architecture. http://formandcode.com/code-examples/simulate-dla
#
# -----------------------------------------------------------------------------
