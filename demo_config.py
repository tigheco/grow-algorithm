"""
GROW: Demo Configuration

author: Tighe Costa
email: tighe.costa@gmail.com
"""

config = {
    "width": 540,                               # environment width
    "height": 540,                              # environment height
    "maxIter": 50,                              # timeout iterations
    "seeds": 15,                                # number of seed cells
    "foodFile": "foodMaps-04.png",              # food map file path
    "mapFile": "foodMaps-00.png",               # area map file path
    "mixRatios": [5, 4, 7],                     # species probability ratios
    "cellTypes": [                              # species properties
        {
         "species": 1,
         "proliferation rate": 1,
         "metabolism": 10,
         "abundance": 1,
         "food to divide": 10*8,
         "food to move": 10*6,
         "division recovery time": 8,
         "food to survive": 10*2,
         "endurance": 180,
        },
        {
         "species": 2,
         "proliferation rate": 1,
         "metabolism": 15,
         "abundance": 1,
         "food to move": 50,
         "food to divide": 50*2,
         "division recovery time": 2,
         "food to survive": 50*2,
         "endurance": 160,
        },
        {
         "species": 3,
         "proliferation rate": 1,
         "metabolism": 20,
         "abundance": 1,
         "food to move": 20*10,
         "food to divide": 20*12,
         "division recovery time": 12,
         "food to survive": 20*2,
         "endurance": 200,
        }
    ],
    "outputSize": (2160, 2160),                 # rendered video size in px
    "outputPath": "_sims/",                     # path to save data and videos
    "timeWarpFactor": 2,                        # render playback factor
}
