import sys
from rng_seeds import *


POPULATION_SIZE = 5
NUMBER_OF_ITERATIONS = 100
ELITISM = 1 #number of individuals to copy to the next generation
TOURNAMENT = 3
PROB_CROSSOVER = 0.9
PROB_MUTATION = (lambda mapped : 1.0 / mapped)
RUN = len(sys.argv) > 1 and int(sys.argv[1]) or 0
SEED = seeds[RUN]
ADD_PHENOTYPE_TO_JSON_OBJECT = False
sampling_snap = [0,10,20,30,40,50,60,70,80,90]
