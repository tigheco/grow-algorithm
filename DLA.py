# -----------------------------------------------------------------------------
# Diffusion-Limited Aggregation: 2D Growth Array in Python
# creator: jocos
# created: 11/03/2016
# modified: 12/13/2016
# -----------------------------------------------------------------------------

import numpy as np
import sys
import imageio
from PIL import Image

from cell import Environment, Tissue


def initialize(width, height, species, seeds, foodFile, mixRatios):
    # build food map from file
    img = Image.open(foodFile)
    img = img.convert(mode='L')
    img = img.resize((width, height), resample=Image.BILINEAR)
    food = np.reshape(np.array(list(img.getdata())), (height, width))

    # initialize environment
    env = Environment(width, height, food, mixRatios)
    env.addSpecies(species)

    # initialize tissues
    tissues = env.addTissues(seeds)

    return env, tissues


def draw(frames, fileName, fieldSize, dispSize):

    imageio.mimwrite(fileName, frames, macro_block_size=8, fps=30)

    return


def main():

    np.random.seed()

    # -------------------------------------------------------------------------
    # user controls
    width = 120                            # environment width
    height = 120                           # environment height
    maxIter = 100                           # timeout iterations
    seeds = 2                              # number of seed cells
    foodFile = 'food/foodMaps-04.png'      # food map file name
    mixRatios = [1, 1, 1]                   # species probability ratios
    species = [                             # species properties
        {
         "id": 1,
         "proliferation rate": 3,
         "metabolism": 2
        },
        {
         "id": 2,
         "proliferation rate": 3,
         "metabolism": 2
        },
        {
         "id": 3,
         "proliferation rate": 3,
         "metabolism": 2
        }
    ]
    # -------------------------------------------------------------------------

    print ('Initializing.........')
    env, tissues = initialize(width, height, species, seeds, foodFile,
                              mixRatios)

    print ('')
    print ('')
    print ('Growing..............')
    framesS = []
    framesT = []
    framesN = []
    framesSi = []
    for i in xrange(maxIter):
        prevField = env.field[1, :, :].copy()

        if i % 1 is 0:
            framesT.append(
                (env.field[0, :, :]*255/len(env.tissuesList)).astype('uint8'))
            framesS.append(
                (env.field[1, :, :]*255/len(species)).astype('uint8'))
            framesN.append(
                (np.maximum(env.field[2, :, :]*255/100,
                            np.zeros((height, width)))).astype('uint8'))
            # framesSi.append(
            #     (-env.field[1, 3:-3, 3:-3]*255/len(species) + 255).astype('uint8'))

        for tissue in tissues:
            tissue.update()

        progress = np.round(i*78/maxIter)
        sys.stdout.write("\r"+"["+"-"*progress+" "*(77-progress)+"]")
        sys.stdout.flush()

        if (env.field[1, :, :] == prevField).all():
            break

    print ('')
    print ('')
    print ('Saving...............')

    # draw(framesT, 'tissues.gif', (width, height), (1000, 1000))
    # draw(framesS, 'species.gif', (width, height), (1000, 1000))
    # draw(framesN, 'nutrients.gif', (width, height), (1000, 1000))
    # draw(framesSi, 'speciesi.gif', (width, height), (1000, 1000))

    draw(framesT, 'tissues.mp4', (width, height), (1920, 1080))
    draw(framesS, 'species.mp4', (width, height), (1920, 1080))
    draw(framesN, 'nutrients.mp4', (width, height), (1920, 1080))
    # draw(framesSi, 'speciesi.mp4', (width, height), (1920, 1080))

    return

if __name__ == "__main__":
    main()

# -----------------------------------------------------------------------------
# references:
# Reas, Casey. "Simulate: Diffusion-Limited Aggregation." Form+Code in Design,
#     Art, and Architecture. http://formandcode.com/code-examples/simulate-dla
# -----------------------------------------------------------------------------
