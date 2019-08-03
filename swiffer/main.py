#!/usr/bin/env python3

"""
GROW: Biologically Inspired Cellular Growth Algorithm

author: Tighe Costa
email: tighe.costa@gmail.com
"""

import numpy as np
import sys

import cv2
import imageio
from PIL import Image

import grow

def initialize(width, height, cellTypes, seeds, foodFile, mixRatios):
    # initialize environment
    food = cv2.imread(foodFile, cv2.IMREAD_GRAYSCALE)
    env = grow.Dish(width, height, food, mixRatios)
    env.addSpecies(cellTypes)

    # initialize tissues
    tissues = env.addTissues(seeds)

    return env, tissues


def draw(frames, fileName, fieldSize, dispSize):
    outFrames = []
    for frame in frames:
        outFrames.append(cv2.resize(frame, dispSize))

    imageio.mimwrite(fileName, outFrames, macro_block_size=16, fps=30, quality=8)

    return


def main():

    np.random.seed()

    print("GROW: Biologically Inspired Cellular Growth Algorithm\n")

    # -------------------------------------------------------------------------
    # user controls
    width = 160                            # environment width
    height = 160                           # environment height
    maxIter = 100                          # timeout iterations
    seeds = 1                            # number of seed cells
    foodFile = "../_food/foodMaps-00.png"      # food map file name
    mixRatios = [1, 1, 1]                   # species probability ratios
    cellTypes = [                             # species properties
        {
         "species": 1,
         "proliferation rate": 1,
         "metabolism": 20,
         "abundance": 1,
         "food to divide": 100
        },
        # {
        #  "species": 2,
        #  "proliferation rate": 1,
        #  "metabolism": 20,
        #  "abundance": 1
        # }
    ]
    outputSize = (800, 800)
    # -------------------------------------------------------------------------

    print("[1/3] Initializing...")
    env, tissues = initialize(width, height, cellTypes, seeds, foodFile,
                              mixRatios)
    print("Complete.")

    print("\n[2/3] Growing...")
    framesS = []
    framesT = []
    framesN = []
    framesSi = []
    for i in range(maxIter):
        prevField = env.links.copy()

        # if i % 1 is 0:
        framesT.append(
            (env.links*255/len(env.tissuesList)).astype("uint8"))
        framesS.append(
            (env.species*255/len(cellTypes)).astype("uint8"))
        framesN.append(
            (np.maximum(env.food*255/100,
                        np.zeros((height, width)))).astype("uint8"))

        for tissue in tissues:
            tissue.update()

        progress = int(i*78/(maxIter-1))
        sys.stdout.write("\r"+"["+"-"*progress+" "*(77-progress)+"]")
        sys.stdout.flush()

        # if (env.links == prevField).all():
        #     print("\nConverged to steady state. Terminating growth.")
        #     break

        if i == maxIter-1:
            print("\nComplete.")

    print("\n[3/3] Saving...")

    draw(framesT, "tissues.mp4", (width, height), outputSize)
    draw(framesS, "species.mp4", (width, height), outputSize)
    draw(framesN, "nutrients.mp4", (width, height), outputSize)

    print("Complete.")

    return

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# references:
# Reas, Casey. "Simulate: Diffusion-Limited Aggregation." Form+Code in Design,
#     Art, and Architecture. http://formandcode.com/code-examples/simulate-dla
# -----------------------------------------------------------------------------
