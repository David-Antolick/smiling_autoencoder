#!/usr/bin/env python3
'''Randomize smiles'''
from openbabel import pybel
import sys, gzip

with gzip.open(sys.argv[1],'rt') as f:
    for line in f:
        line = line.rstrip()
        m = pybel.readstring('smi',line)
        random = m.write('smi',opt={'k':None,'C':True}).rstrip()  #kekulized and non-canonical random
        if len(random) < 150 and '%' not in random: #randomizing can result in longer strings or have characters not in our reduced set
            print(random)
        elif len(line) < 150: #so output canonical which should be right size
            print(line)