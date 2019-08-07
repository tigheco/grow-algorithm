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


def initialize(width, height, cellTypes, seeds, foodFile, mapFile, mixRatios):
    # initialize environment
    food = cv2.imread(foodFile, cv2.IMREAD_GRAYSCALE)
    map = cv2.imread(mapFile, cv2.IMREAD_GRAYSCALE)
    env = grow.Dish(width, height, food, map, mixRatios)
    env.addSpecies(cellTypes)

    # initialize tissues
    tissues = env.addTissues(seeds)

    return env, tissues


def draw(frames, fileName, fieldSize, dispSize):
    outFrames = []
    for frame in frames:
        outFrames.append(cv2.resize(frame, dispSize))

    imageio.mimwrite(fileName, outFrames, macro_block_size=16, fps=30, quality=8)

    return None


def main():

    np.random.seed()

    print("GROW: Biologically Inspired Cellular Growth Algorithm\n")

    # -------------------------------------------------------------------------
    # user controls
    width = 640                                 # environment width
    height = 360                                # environment height
    maxIter = 500                               # timeout iterations
    seeds = int(width/4)                        # number of seed cells
    foodFile = "../_food/foodMaps-00.png"       # food map file name
    mapFile =  "../_food/title-03.png"
    mixRatios = [1, 1, 1]                       # species probability ratios
    cellTypes = [                               # species properties
        {
         "species": 1,
         "proliferation rate": 1,
         "metabolism": 5,
         "abundance": 1,
         "food to divide": 5*5,
         "division recovery time": 10,
         "food to survive": 5*2,
         "endurance": 100,
        },
        {
         "species": 2,
         "proliferation rate": 1,
         "metabolism": 7,
         "abundance": 1,
         "food to divide": 7*5,
         "division recovery time": 10,
         "food to survive": 7*2,
         "endurance": 100,
        },
        {
         "species": 3,
         "proliferation rate": 1,
         "metabolism": 9,
         "abundance": 1,
         "food to divide": 9*4,
         "division recovery time": 10,
         "food to survive": 9*2,
         "endurance": 100,
        }
    ]
    outputSize = 1280, 720
    # -------------------------------------------------------------------------

    print("[1/3] Initializing...")
    env, tissues = initialize(width, height, cellTypes, seeds, foodFile,
                              mapFile, mixRatios)
    print("Complete.")

    print("\n[2/3] Growing...")
    framesSpecies = []
    framesLinks = []
    framesFood = []
    framesFoodSums = []
    for i in range(1, maxIter+1):
        prevFood = env.food.copy()
        prevLinks = env.links.copy()

        # if i % 1 is 0:
        framesLinks.append(
            (env.links*255/len(env.tissuesList)).astype("uint8"))
        framesSpecies.append(
            (env.species*255/len(cellTypes)).astype("uint8"))
        framesFood.append(
            (np.maximum(env.food*255/100,
                        np.zeros((height, width)))).astype("uint8"))
        framesFoodSums.append(
            (np.maximum(env.foodSums*255/4900,
                        np.zeros((height+2, width+2)))).astype("uint8"))

        if (env.nCells == 0):
            print("\nAll cells dead in %i iterations. Terminating simulation." % i)
            break

        # if (env.links == prevLinks).all():
        #     print("\nConverged to steady state in %i iterations. Terminating growth." % i)
        #     break

        # if (env.food == prevFood).all():
        #     print("\nConsumed all nutrients in %i iterations. Terminating growth." % i)
        #     break

        for tissue in tissues:
            tissue.update()

        progress = int(i*78/(maxIter-1))
        sys.stdout.write("\r"+"["+"-"*progress+" "*(77-progress)+"]")
        sys.stdout.flush()

        if i == maxIter:
            print("\nCompleted %i iterations." % i)

    print("\n[3/3] Saving...")

    # draw(framesLinks, "tissues.mp4", (width, height), outputSize)
    draw(framesSpecies, "species.mp4", (width, height), outputSize)
    draw(framesFood, "nutrients.mp4", (width, height), outputSize)
    # draw(framesFoodSums, "nutrientSums.mp4", (width+2, height+2), outputSize)

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
