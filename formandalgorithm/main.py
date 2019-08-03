#!/usr/bin/env python3

"""
GROW: Biologically Inspired Cellular Growth Algorithm

author: Tighe Costa
email: tighe.costa@gmail.com
created: 2016/11/03
modified: 2019/07/27
"""

import numpy as np
import sys
import imageio
from PIL import Image

import grow

def initialize(width, height, cellTypes, seeds, foodFile, mixRatios):
    # build food map from file
    img = Image.open(foodFile)
    img = img.convert(mode="L")
    img = img.resize((width, height), resample=Image.BILINEAR)
    food = np.reshape(np.array(list(img.getdata())), (height, width))

    # initialize environment
    env = grow.Environment(width, height, food, mixRatios)
    env.addSpecies(cellTypes)

    # initialize tissues
    tissues = env.addTissues(seeds)

    return env, tissues


def draw(frames, fileName, fieldSize, dispSize):
    # TODO (Tighe) scale fieldSize to dispSize using np.kron
    imageio.mimwrite(fileName, frames, macro_block_size=16, fps=30)

    return


def main():

    np.random.seed()

    print("GROW: Biologically Inspired Cellular Growth Algorithm\n")

    # -------------------------------------------------------------------------
    # user controls
    width = 160                            # environment width
    height = 160                           # environment height
    maxIter = 100                           # timeout iterations
    seeds = 4                              # number of seed cells
    foodFile = "../_food/foodMaps-04.png"      # food map file name
    mixRatios = [1, 1, 1]                   # species probability ratios
    cellTypes = [                             # species properties
        {
         "species": 1,
         "proliferation rate": 3,
         "metabolism": 2,
         "abundance": 1
        },
        {
         "species": 2,
         "proliferation rate": 3,
         "metabolism": 2,
         "abundance": 1
        },
        {
         "species": 3,
         "proliferation rate": 3,
         "metabolism": 2,
         "abundance": 1
        }
    ]
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
        prevField = env.field[1,:,:].copy()

        if i % 1 is 0:
            framesT.append(
                (env.field[0,:,:]*255/len(env.tissuesList)).astype("uint8"))
            framesS.append(
                (env.field[1,:,:]*255/len(cellTypes)).astype("uint8"))
            framesN.append(
                (np.maximum(env.field[2,:,:]*255/100,
                            np.zeros((height, width)))).astype("uint8"))
            # framesSi.append(
            #     (-env.field[1, 3:-3, 3:-3]*255/len(cellTypes) + 255).astype("uint8"))

        for tissue in tissues:
            tissue.update()

        progress = int(i*78/(maxIter-1))
        sys.stdout.write("\r"+"["+"-"*progress+" "*(77-progress)+"]")
        sys.stdout.flush()

        if (env.field[1,:,:] == prevField).all():
            print("\nConverged to steady state. Terminating growth.")
            break

        if i == maxIter-1:
            print("\nComplete.")

    print("\n[3/3] Saving...")

    draw(framesT, "tissues.mp4", (width, height), (1600, 1600))
    draw(framesS, "species.mp4", (width, height), (1600, 1600))
    draw(framesN, "nutrients.mp4", (width, height), (1600, 1600))
    # draw(framesSi, 'speciesi.mp4', (width, height), (1920, 1080))

    print("Complete.")

    return

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# references:
# Reas, Casey. "Simulate: Diffusion-Limited Aggregation." Form+Code in Design,
#     Art, and Architecture. http://formandcode.com/code-examples/simulate-dla
# -----------------------------------------------------------------------------
