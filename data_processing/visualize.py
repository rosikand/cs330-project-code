import numpy as np 
import pickle
import pdb
from rsbox import ml 



cell_types = ['shsy5y', 'skbr3', 'a172', 'skov3', 'bt474', 'huh7', 'mcf7', 'bv2']


for cell in cell_types:
    in_file = open("dists/" + cell + "_dist.pkl", "rb")
    rd = pickle.load(in_file)
    print(len(rd))
    print(cell)
    ml.plot(rd[0][0], color=False)
    ml.plot(rd[0][1], color=False)
    print('-'*20)
