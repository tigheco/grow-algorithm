"""
GROW: Back End

author: Tighe Costa
email: tighe.costa@gmail.com
"""

import numpy as np
import scipy.signal as sig
import scipy.stats as st
import line_profiler

import cv2
import random
import math
from operator import add
import copy

from PIL import Image


indices = np.array([[[0,0], [0,1], [0,2]],
                    [[1,0], [1,1], [1,2]],
                    [[2,0], [2,1], [2,2]]])


class Dish():
    """
    Everything happens in the dish.

    map defines the space
    species keeps track of where cells are by type
    food keeps track of the available nutrients in that space
    links keeps track of where cells are by connection
    """

    map = []                    # map of space (1 = valid, 0 = invalid)
    species = []                # map of cell types
    food = []                   # map of nutrients at each location
    links = []                  # map of cell connections

    foodSums = []               # map of nutrients within reach of each location
    biteR = 1                   # radius of bite
    biteKern = np.ones((2*biteR+1,2*biteR+1))   # kernel defining reach of a bite

    width = []                  # field width
    height = []                 # field height
    speciesList = []            # species list
    tissuesList = []            # tissues list
    mixRatios = []              # mix ratios

    def __init__(self, width, height, foodImg, mixRatios):
        # initialize space map
        # TODO: make map from image
        # TODO: make map true size of frame
        Dish.map = np.pad(np.ones((height-2, width-2)),
                          (1, 1), 'constant', constant_values=0)

        # initialize cell type map
        Dish.species = np.zeros((height, width), dtype=np.int8)

        # build food map from file
        foodImg = cv2.resize(foodImg, (width, height))
        Dish.food = foodImg / 255.0 * 100

        # build food within reach map from file
        Dish.foodSums = sig.convolve(Dish.food.copy(), Dish.biteKern)

        # initialize links map
        Dish.links = np.zeros((height, width), dtype=np.int8)

        Dish.width = width
        Dish.height = height

        return

    # TODO: combine into init
    def addSpecies(self, species):
        # save cell types to dish
        Dish.speciesList = species

        # pull abundance ratios from cell types and normalize sum to 1
        Dish.mixRatios = normSum([x["abundance"] for x in species])

    def addTissues(self, quantity):
        valid_seeds = np.logical_and(Dish.map == 1, Dish.food > 0)

        idy, idx = np.where(valid_seeds)

        for n in range(quantity):
            # randomly choose location
            ind = np.random.randint(0, len(idx))
            seed = [idx[ind], idy[ind]]

            # pick species
            species = np.random.choice(Dish.speciesList, p=Dish.mixRatios)

            # create seed cell
            cell = Cell(Dish.links, species["species"], seed)

            # create tissue
            Dish.tissuesList.append(
                Tissue(Dish, species, [cell], seed, n+1)
            )

            # populate maps
            Dish.links[seed[1], seed[0]] = n+1
            Dish.species[seed[1], seed[0]] = species["species"]

        return Dish.tissuesList


class Tissue():

    env = []
    stepDist = np.array([1/np.sqrt(2), 1, 1/np.sqrt(2),
                         1,            0, 1,
                         1/np.sqrt(2), 1, 1/np.sqrt(2)])

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
        self.divThresh = species["food to divide"]
        self.divRecover = species["division recovery time"]

    @profile
    def update(self):
        self.age += 1

        # handle proliferation rates
        for rate in range(self.proliferationRate):
            # update cells
            for cell in list(self.cells):
                # feed the cell
                self.feed(cell)

                # continue to next cell if done dividing
                if cell.done is True:
                    continue
                # let the cell rest if it needs to
                elif cell.resting:
                    cell.timer += 1

                    # reset after enough rest
                    if cell.timer > self.divRecover:
                        cell.resting = False
                        cell.timer = 0

                # but once its done resting do some stuff
                else:
                    # divide if possible
                    if cell.food > self.divThresh:
                        self.divide(cell)

                    # otherwise consider moving
                    else:
                        self.move(cell)

                cell.age += 1       # increment age counter
        return

    @profile
    def feed(self, cell):
        """
        Consume nutrients from location and neighbors in proportion to
        availability.
        """
        # pull available nutrients
        nutrients = get_neighbors(Dish.food, cell.x, cell.y, 1, True)

        # feed if there are nutrients
        if np.sum(nutrients) > self.metabolism:
            # ignore spots with nutrients < 0
            nutrients[nutrients < 0] = 0
            # distribute metabolism proportional to available food
            # bite = (self.metabolism * normSum(nutrients)).reshape(3,3)
            bite = self.metabolism * normSum(nutrients)
            # take bite out of food
            Dish.food[cell.y-1:cell.y+2, cell.x-1:cell.x+2] += -bite

            # # take bite out of foodSums
            # biteSums = sig.convolve(bite, Dish.biteKern)
            # r = Dish.biteR
            # Dish.foodSums[cell.y-r-1:cell.y+r+2, cell.x-r-1:cell.x+r+2] += -biteSums

            # add to cell's food
            cell.food += self.metabolism

        return

    @profile
    def move(self, cell):
        """
        Move cell to new location. Does not move cell if currently at best
        location or there are no available locations to move.
        """
        # get relative movement to make
        delta = self.get_step(cell, forceMove=False)

        # throw error if no move available
        if np.isnan(delta).any():
            raise Exception("get_step should always return valid delta for move.")

        # remove old location from maps of cell types and connections
        Dish.links[cell.y, cell.x] = 0
        Dish.species[cell.y, cell.x] = 0

        # update location of cell
        cell.x += delta[1]
        cell.y += delta[0]

        # update maps of cell types and connections
        Dish.links[cell.y, cell.x] = self.index
        Dish.species[cell.y, cell.x] = self.species

        return

    @profile
    def divide(self, cell):
        """
        Spawn new cell in new location. Does nothing if there are no available
        locations to spawn.
        """
        # get relative location of new cell
        delta = self.get_step(cell, forceMove=True)

        # if all surrounding cells are occupied
        if np.isnan(delta).any():
            cell.done = True
        # if there is a valid step that can be made
        else:
            # calculate absolute location of new cell
            new_loc = [cell.x + delta[1], cell.y + delta[0]]

            # create cell at new location
            self.cells.append(
                Cell(Dish.links, self.species, (new_loc[0], new_loc[1]))
            )

            # update maps of cell types and connections
            Dish.links[new_loc[1], new_loc[0]] = self.index
            Dish.species[new_loc[1], new_loc[0]] = self.species

            self.nCells += 1
            cell.food += -self.divThresh
            cell.resting = True

        # # if next to a different tissue of the same species
        # elif status[0] == "merge tissue":
        #     # get tissue from list
        #     tissue = [tissue for tissue in Dish.tissuesList
        #               if tissue.index == status[1]][0]
        #     self.merge(tissue)
        #     return
        #
        # elif status[0] == "no growth":
        #     cell.dividing = False
        #     return

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

        return

    def reset(self):
        # randomly choose a cell of your Tissue
        return np.random.choice(self.cells)

    @profile
    def get_step(self, cell, forceMove):
        """
        get_step looks through a cell's neighbors and returns the best step
        that can be made by that cell.

        input
            cell: cell to look at (Cell object)
            forceMove: include option of staying still (bool)
        output
            delta: best relative movement [delta Y, delta X] (np.array)
                   returns [nan, nan] if no steps possible
        """
        # get valid locations
        valid_spaces = get_neighbors(Dish.map, cell.x, cell.y, 1, True)

        # get unoccupied locations
        full_spaces = get_neighbors(Dish.species, cell.x, cell.y, 1, True)
        # include center location if move is not forced
        if forceMove is False:
            # full_spaces[4] = 0
            full_spaces[1,1] = 0

        spaces = np.logical_and(valid_spaces==1, full_spaces==0)

        # nan if there are no possible steps
        if not np.any(spaces):
            delta = [np.nan, np.nan]

        # best step otherwise
        else:
            # # get nutrients available to neighbors
            # nutrients = get_neighbors(Dish.foodSums,
            #                          cell.x+Dish.biteR, cell.y+Dish.biteR,
            #                          1, True)

            # get nutrients at neighbors
            nutrients = get_neighbors(Dish.food, cell.x, cell.y, 1, True)

            # nutrients at possible locations
            # valid locations == 1, unoccupied spaces == 0
            steps = nutrients*spaces

            # find steps with the most food
            # gets array indices for steps with max food
            # translates absolute steps to steps relative to center with -1
            best_steps = indices[steps == np.max(steps)] - 1

            # if multiple options
            if len(best_steps) > 1:
                delta = random.choice(best_steps)
            else:
                delta = best_steps[0]

        return delta

    def get_state(self, cell):
        """
        get_state
        """
        status = self.get_step(cell, forceMove=True)

        if status[0] == "no moves":
            return ("no moves",)

        # if along edge, alone
        if Dish.map[cell.p[1], cell.p[0]] == 0:
            return ("along edge",)

        # pull neighbors
        neighborsT = get_neighbors(Dish.links, cell.p[0], cell.p[1], 1)
        neighborsS = get_neighbors(Dish.species, cell.p[0], cell.p[1], 1)

        # check neighbors tissue
        likeT = [x for x in neighborsT if x == self.index]
        unlikeT = [x for x in neighborsT if x != 0 and x != self.index]

        # check neighbors species
        likeS = [x for x in neighborsS if x == self.species]
        unlikeS = [x for x in neighborsS if x != 0 and x != self.species]

        # if a neighbor is of the same tissue
        if len(likeT) > 0 and len(unlikeT) == 0:
            return ("grow tissue",)

        # if a neighbor is of the same species
        elif len(likeS) > 0 and len(unlikeS) == 0:

            index = unlikeT[0]

            if index > self.index:
                return ("merge tissue", index)

        # if a neighbor is of a different species
        return ("no growth",)

    # stepRandom obsolete
    def stepRandom(self, cell):
        """
        Steps cell probe to step in a random direction.
        """
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

    # updateCenter obsolete
    def updateCenter(self):
        idy, idx = np.where(Dish.links == self.index)

        cx = float(sum(idx)) / float(len(idx))
        cy = float(sum(idy)) / float(len(idy))

        self.center[0] = int(np.random.choice([np.floor(cx)-1, np.ceil(cx)+1]))
        self.center[1] = int(np.random.choice([np.floor(cy)-1, np.ceil(cy)+1]))

        self.center[0] = int(cx)
        self.center[1] = int(cy)

        return

class Cell():
    """
    Cell class
    """

    # @profile
    def __init__(self, map, species, position, dividing=True):
        self.species = species
        self.x = position[0]
        self.y = position[1]

        self.age = 0
        self.food = 0

        self.dividing = dividing
        self.resting = False
        self.timer = 0

        self.done = False


        # TODO: get rid of this dumb ass probe variable
        self.p = [float('NaN'),float('NaN')]

        # # vectorizing locations addition 2019/08/01
        # self.loc = np.zeros(map.shape, dtype=bool)
        # self.loc[..., position[1], position[0]] = 1
        #
        # # vectorizing neighbors addition 2019/08/01
        # self.neighbors = np.zeros(map.shape, dtype=bool)
        # self.neighbors[..., position[1]-1:position[1]+2, position[0]-1:position[0]+2] = 1
        # # self.neighbors[..., position[0], position[1]] = 0

def normpdf(x, mu=0, sigma=1):
    """
    Normal probability density function. See MATLAB documentation for normpdf.
    """
    y = []
    for n in x:
        u = (n-mu)/abs(sigma)
        y.append((1/(np.sqrt(2*np.pi)*abs(sigma)))*np.exp(-u*u/2))
    return y

def normSum(array, sum=1):
    """
    Normalize input array to sum.
    """
    total = np.sum(array)

    if total == 0:
        return array
    else:
        return array/total

def gkern(kernlen=3, nsig=3):
    """
    Returns a 2D Gaussian kernel.
    """
    x = np.linspace(-nsig, nsig, kernlen+1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d/kern2d.sum()

@profile
def get_neighbors(array, cx, cy, r, includeCenter=False):
    """
    Returns values of neighbors in array of location cx, cy as 1D numpy
    array.
    """
    return array[cy-r:cy+r+1, cx-r:cx+r+1]

