"""
GROW: Back End

author: Tighe Costa
email: tighe.costa@gmail.com
"""

import numpy as np
import line_profiler

import random
import math
from operator import add
import copy

from PIL import Image

class Dish():
    """
    Everything happens in the dish.

    map defines the space and keeps track of where cells are
    food keeps track of the available nutrients in that space
    links keeps track of which cells are connected
    """

    map = []                # map of where cells are
    food = []               # map of available nutrients
    links = []              # map of cell connections

    edges = []              # edge map
    width = []              # field width
    height = []             # field height
    speciesList = []        # species list
    tissuesList = []        # tissues list
    mixRatios = []          # mix ratios

    def __init__(self, width, height, foodImg, mixRatios):
        # initialize spatial map
        Dish.map = np.zeros((height, width), dtype=np.int8)

        # build food map from file
        foodImg = foodImg.convert(mode="L")
        foodImg = foodImg.resize((width, height), resample=Image.BILINEAR)
        Dish.food = np.reshape(np.array(list(foodImg.getdata())), (height, width))
        Dish.food = Dish.food / 255.0 * 100

        # initialize links map
        Dish.links = np.zeros((height, width), dtype=np.int8)

        # old initializations
        Dish.edges = np.pad(np.zeros((height-6, width-6)),
                                   (3, 3),
                                   'constant', constant_values=1)
        Dish.width = width
        Dish.height = height
        # Dish.mixRatios = [x/float(sum(mixRatios)) for x in mixRatios]
        return

    def addSpecies(self, species):
        # save cell types to dish
        Dish.speciesList = species

        # pull abundance ratios from cell types
        Dish.mixRatios = [x["abundance"] for x in species]
        # normalize abundance ratios
        Dish.mixRatios = [float(i)/sum(Dish.mixRatios) for i in Dish.mixRatios]

    def addTissues(self, quantity):
        idy, idx = np.where(Dish.food > 0)

        for n in range(quantity):
            # generate random seed
            ind = np.random.randint(0, len(idx))
            seed = [idx[ind], idy[ind]]

            if (Dish.links[seed[1], seed[0]] == 0).all():
                # pick species
                species = np.random.choice(Dish.speciesList,
                                           p=Dish.mixRatios)

                # create seed cell
                cell = Cell(Dish.links, species["species"], seed)

                # create tissue
                Dish.tissuesList.append(
                    Tissue(Dish, species, [cell], seed, n+1)
                )

                # populate maps
                Dish.links[seed[1], seed[0]] = n+1
                Dish.map[seed[1], seed[0]] = species["species"]

        return Dish.tissuesList


class Tissue():

    env = []

    def __init__(self, environment, species, cells, center, index):
        # characteristics
        Dish = environment
        self.age = 0
        self.species = species["species"]
        self.nCells = len(cells)
        self.center = center
        self.cells = cells
        self.index = index

        # properties
        self.proliferationRate = species["proliferation rate"]
        self.metabolism = species["metabolism"]

    def update(self):
        self.age += 1

        # handle proliferation rates
        for rate in range(self.proliferationRate):

            # update cells
            for cell in list(self.cells):
                if cell.dividing:
                    cell.age += 1
                    self.feed(cell)
                    self.divide(cell)
        return

    @profile
    def feed(self, cell):
        # pull available nutrients
        nutrients = getNeighbors(Dish.food, cell.x, cell.y, 1, True)

        # feed if there are nutrients
        if sum(nutrients) > self.metabolism:
            bite = self.metabolism / float(np.count_nonzero(nutrients))
            Dish.food[cell.y-1:cell.y+2, cell.x-1:cell.x+2] += -bite
        # stop dividing otherwise
        else:
            cell.dividing = False
        # # kill if there are no nutrients
        # else:
        #     Dish.links[cell.y, cell.x] = 0
        #     Dish.map[cell.y, cell.x] = 0
        #     # Dish.food[, cell.y, cell.x] = cell.age
        #     self.cells.remove(cell)
        #     self.nCells += -1

    def divide(self, cell):
        self.step(cell)

        if (Dish.links[cell.p[1], cell.p[0]] != 0).all():
            # self.reset()
            return

        status = self.alone(cell)

        # if along edge
        if status[0] == "along edge":
            cell.dividing = False
            return

        # if next to a cell of the same tissue
        elif status[0] == "grow tissue":
            Dish.links[cell.p[1], cell.p[0]] = self.index
            Dish.map[cell.p[1], cell.p[0]] = self.species

            self.cells.append(
                Cell(Dish.links, self.species, (cell.p[0], cell.p[1]))
            )

            self.nCells += 1
            return

        # if next to a different tissue of the same species
        elif status[0] == "merge tissue":
            # get tissue from list
            tissue = [tissue for tissue in Dish.tissuesList
                      if tissue.index == status[1]][0]
            self.merge(tissue)
            return

        elif status[0] == "no growth":
            cell.dividing = False
            return

    def merge(self, tissue):
        """
        Merge with a tissue of the same species.
        """
        # get new index
        self.index = self.index if self.index < tissue.index else tissue.index

        # add cells
        self.nCells += tissue.nCells
        self.cells.extend(tissue.cells)

        # update maps
        for cell in tissue.cells:
            Dish.links[cell.y, cell.x] = self.index

        # get new center
        self.updateCenter()

        # delete other tissue
        Dish.tissuesList.remove(tissue)

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
                    p[0] < Dish.width and
                    p[1] < Dish.height):
                cell.p = p
                break
        return

    def updateCenter(self):
        idy, idx = np.where(Dish.links == self.index)

        cx = float(sum(idx)) / float(len(idx))
        cy = float(sum(idy)) / float(len(idy))

        self.center[0] = int(np.random.choice([np.floor(cx)-1, np.ceil(cx)+1]))
        self.center[1] = int(np.random.choice([np.floor(cy)-1, np.ceil(cy)+1]))

        self.center[0] = int(cx)
        self.center[1] = int(cy)

        return

    def alone(self, cell):
        # if along edge, alone
        if Dish.edges[cell.p[1], cell.p[0]] == 1:
            return ("along edge",)

        # pull neighbors
        neighborsT = getNeighbors(Dish.links,
                                       cell.p[0], cell.p[1], 1)
        neighborsS = getNeighbors(Dish.map,
                                       cell.p[0], cell.p[1], 1)

        # check neighbors tissue
        likeT = [x for x in neighborsT if x == self.index]
        unlikeT = [x for x in neighborsT if x != 0 and x != self.index]

        # check neighbors species
        likeS = [x for x in neighborsS if x == self.species]
        unlikeS = [x for x in neighborsS if x != 0 and x != self.species]

        # pull available nutrients
        nutrients = getNeighbors(Dish.food,
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


class Cell():
    """
    Cell class
    """

    # @profile
    def __init__(self, map, species, position, dividing=True):
        self.species = species
        self.x = position[0]
        self.y = position[1]
        self.dividing = dividing
        self.p = [-1, -1]
        self.age = 0
        self.health = 10

        # vectorizing locations addition 2019/08/01
        self.loc = np.zeros(map.shape, dtype=bool)
        self.loc[..., position[0], position[1]] = 1

        # vectorizing neighbors addition 2019/08/01
        self.neighbors = np.zeros(map.shape, dtype=bool)
        self.neighbors[..., position[0]-1:position[0]+2, position[1]-1:position[1]+2] = 1
        # self.neighbors[..., position[0], position[1]] = 0

def normpdf(x, mu, sigma):
    """
    Normal probability density function. See MATLAB documentation for normpdf.
    """
    y = []
    for n in x:
        u = (n-mu)/abs(sigma)
        y.append((1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2))
    return y

def getNeighbors(array, cx, cy, r, includeCenter=False):
    """
    Returns values of neighbors in array of location cx, cy as 1D numpy
    array. Removes value of location if includeCenter=False.
    """
    # pull section of field centered around cx, cy with reach r
    subsample = array[cy-r:cy+r+1, cx-r:cx+r+1].ravel()

    if includeCenter:
        # do nothing
        return subsample
    else:
        # pop out field[cx, cy] value
        return np.delete(subsample, int(len(subsample)/2))
