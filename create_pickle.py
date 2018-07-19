import argparse
import pickle
import os
import sys
import graph

def pickle_graphs(filename, NUM, beta):
    graphs = []
    for i in range(NUM):
        graphs.append(graph.RandomGraph(n=100))
        graphs.append(graph.PlantedClique(n=100, beta=beta))
    with open(filename, mode='wb') as f:
        pickle.dump(graphs, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--beta', type=float, default=1.0)
    parser.add_argument('-n', '--num', type=int, default=500)
    args = parser.parse_args()
    filename = 'beta{}_num_{}.pickle'.format(args.beta, args.num)
    if os.path.exists(filename):
        if input('do you want to override {} (y or n)?'.format(filename)) != 'y':
            sys.exit()
    pickle_graphs(filename, args.num, args.beta)
    print('created {}!'.format(filename))
