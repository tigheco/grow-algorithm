"""
GROW: Environment and Cell Classes

author: Tighe Costa
email: tighe.costa@gmail.com
created: 2016/09/30
modified: 2019/07/27
"""

import numpy as np
# import line_profiler

import random
import math
from operator import add
import copy

from PIL import Image


class Environment():

    field = []              # cell field
    edges = []              # edge map
    width = []              # field width
    height = []             # field height
    speciesList = []        # species list
    tissuesList = []        # tissues list
    mixRatios = []          # mix ratios

    def __init__(self, width, height, food, mixRatios):
        Environment.field = np.zeros((3, height, width), dtype=object)
        Environment.field[2, :, :] = food / 255.0 * 100
        Environment.edges = np.pad(np.zeros((height-6, width-6)),
                                   (3, 3),
                                   'constant', constant_values=1)
        Environment.width = width
        Environment.height = height
        Environment.mixRatios = [x/float(sum(mixRatios)) for x in mixRatios]
        return

    def addSpecies(self, species):
        Environment.speciesList = species

    def addTissues(self, quantity):
        idy, idx = np.where(Environment.field[2, :, :] > 0)

        for n in xrange(quantity):
            # generate random seed
            ind = np.random.randint(0, len(idx))
            seed = [idx[ind], idy[ind]]

            if (Environment.field[0:2, seed[1], seed[0]] == 0).all():
                # pick species
                species = np.random.choice(Environment.speciesList,
                                           p=Environment.mixRatios)

                # create seed cell
                cell = Cell(species["id"], seed)

                # create tissue
                Environment.tissuesList.append(
                    Tissue(Environment, species, [cell], seed, n+1)
                )

                # populate field
                Environment.field[0, seed[1], seed[0]] = n+1
                Environment.field[1, seed[1], seed[0]] = species["id"]

        return Environment.tissuesList


class Tissue():

    env = []

    def update(self):
        self.age += 1

        # handle proliferation rates
        for rate in xrange(self.proliferationRate):

            # update cells
            for cell in list(self.cells):
                if cell.dividing:
                    cell.age += 1
                    self.feed(cell)
                    self.divide(cell)
        return

    def feed(self, cell):
        # pull available nutrients
        nutrients = self.getNeighbors(Tissue.env.field[2, :, :],
                                      cell.x, cell.y, 1, True)

        # feed if there are nutrients
        if sum(nutrients) > self.metabolism:
            Tissue.env.field[2, cell.y-1:cell.y+2, cell.x-1:cell.x+2] += (
                - self.metabolism / float(np.nonzero(nutrients)[0].shape[0]))
        # stop dividing otherwise
        else:
            cell.dividing = False
        # # kill if there are no nutrients
        # else:
        #     Tissue.env.field[0, cell.y, cell.x] = 0
        #     Tissue.env.field[1, cell.y, cell.x] = 0
        #     # Tissue.env.field[2, cell.y, cell.x] = cell.age
        #     self.cells.remove(cell)
        #     self.nCells += -1

    def divide(self, cell):
        self.step(cell)

        if (Tissue.env.field[0:2, cell.p[1], cell.p[0]] != 0).all():
            # self.reset()
            return

        status = self.alone(cell)

        # if along edge
        if status[0] == "along edge":
            cell.dividing = False
            return

        # if next to a cell of the same tissue
        elif status[0] == "grow tissue":
            Tissue.env.field[0, cell.p[1], cell.p[0]] = self.index
            Tissue.env.field[1, cell.p[1], cell.p[0]] = self.species

            self.cells.append(
                Cell(self.species, (cell.p[0], cell.p[1]))
            )

            self.nCells += 1
            return

        # if next to a different tissue of the same species
        elif status[0] == "merge tissue":
            # get tissue from list
            tissue = [tissue for tissue in Tissue.env.tissuesList
                      if tissue.index == status[1]][0]
            self.merge(tissue)
            return

        elif status[0] == "no growth":
            cell.dividing = False
            return

    def __init__(self, environment, species, cells, center, index):
        # characteristics
        Tissue.env = environment
        self.age = 0
        self.species = species["id"]
        self.nCells = len(cells)
        self.center = center
        self.cells = cells
        self.index = index

        # properties
        self.proliferationRate = species["proliferation rate"]
        self.metabolism = species["metabolism"]
        self.PCDthreshold = 20

    def merge(self, tissue):
        """
        Merge with a tissue of the same species.
        """
        # get new index
        self.index = self.index if self.index < tissue.index else tissue.index

        # add cells
        self.nCells += tissue.nCells
        self.cells.extend(tissue.cells)

        # update field
        for cell in tissue.cells:
            Tissue.env.field[0, cell.y, cell.x] = self.index

        # get new center
        self.updateCenter()

        # delete other tissue
        Tissue.env.tissuesList.remove(tissue)

    def reset(self):
        # randomly choose a cell of your Tissue
        return np.random.choice(self.cells)

    def step(self, cell):
        # step in random direction
        while True:
            probex = random.choice([-1, 0, 1])
            probey = random.choice([-1, 0, 1])
            p = [cell.x + probex, cell.y + probey]

            # if in bounds
            if (p[0] >= 0 and
                    p[1] >= 0 and
                    p[0] < Tissue.env.width and
                    p[1] < Tissue.env.height):
                cell.p = p
                break
        return

    def updateCenter(self):
        idy, idx = np.where(Tissue.env.field[0, :, :] == self.index)

        cx = float(sum(idx)) / float(len(idx))
        cy = float(sum(idy)) / float(len(idy))

        self.center[0] = int(np.random.choice([np.floor(cx)-1, np.ceil(cx)+1]))
        self.center[1] = int(np.random.choice([np.floor(cy)-1, np.ceil(cy)+1]))

        self.center[0] = int(cx)
        self.center[1] = int(cy)

        return

    def alone(self, cell):
        # if along edge, alone
        if Tissue.env.edges[cell.p[1], cell.p[0]] == 1:
            return ("along edge",)

        # pull neighbors
        neighborsT = self.getNeighbors(Tissue.env.field[0, :, :],
                                       cell.p[0], cell.p[1], 1)
        neighborsS = self.getNeighbors(Tissue.env.field[1, :, :],
                                       cell.p[0], cell.p[1], 1)

        # check neighbors tissue
        likeT = filter(lambda x: x == self.index, neighborsT)
        unlikeT = filter(lambda x: x != 0 and x != self.index, neighborsT)

        # check neighbors species
        likeS = filter(lambda x: x == self.species, neighborsS)
        unlikeS = filter(lambda x: x != 0 and x != self.species, neighborsS)

        # pull available nutrients
        nutrients = self.getNeighbors(Tissue.env.field[2, :, :],
                                      cell.x, cell.y, 1, True)

        # if a neighbor is of the same tissue
        if len(likeT) > 0 and len(unlikeT) == 0 and sum(nutrients) > 0:
            return ("grow tissue",)

        # if a neighbor is of the same species
        elif len(likeS) > 0 and len(unlikeS) == 0 and sum(nutrients) > 0:

            index = unlikeT[0]

            if index > self.index:
                return ("merge tissue", index)

        # if a neighbor is of a different species
        return ("no growth",)

    def getNeighbors(self, field, cx, cy, r, includeCenter=False):
        subsample = field[cy-r:cy+r+1, cx-r:cx+r+1].ravel().tolist()
        if not includeCenter:
            del subsample[(1+2*r)**2 / 2]
        return subsample

    def draw(self, field):
        img = Image.new('L', field.shape)
        pxdata = [px * 255/len(Tissue.env.tissuesList) for px in field.ravel()]
        img.putdata(pxdata)
        img = img.resize((500, 500))
        img.show()
        return


class Cell():

    def __init__(self, species, position, dividing=True):
        self.species = species
        self.x = position[0]
        self.y = position[1]
        self.dividing = dividing
        self.p = [-1, -1]
        self.age = 0
        self.health = 10


def normpdf(x, mu, sigma):
    y = []
    for n in x:
        u = (n-mu)/abs(sigma)
        y.append((1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2))
    return y
