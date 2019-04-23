import numpy as np
import graph

class Multiset:
    def __init__(self, graphIndex, v, neighbors):
        self.graphIndex = graphIndex
        self.v = v
        self.neighbors = neighbors

    def __gt__(self, other):
        return self.neighbors > other.neighbors

    def __str__(self):
        return "{} th graph, {} th vertex, {}".format(self.graphIndex, self.v, self.neighbors)

    def sortNeighbors(self):
        self.neighbors.sort()

def WLKernel(graphs, H):
    vertexSum = sum([g.n for g in graphs])
    kernel = np.zeros((len(graphs), vertexSum * H), dtype=int)
    acc = 0
    for graphIndex, g in enumerate(graphs):
        for v in range(g.n):
            kernel[graphIndex, g.l[0][v]] += 1
            acc = max(acc, g.l[0][v])
    yield kernel[:, np.any(kernel != 0, axis=0)]

    for h in range(1, H):
        print("iteration: {} / {}".format(h, H))
        multisets = []
        for x, g in enumerate(graphs):
            for i in range(g.n):
                multiset = Multiset(x, i, g.l[h-1][g.adjacencyList(i)].tolist())
                multiset.sortNeighbors()
                multisets.append(multiset)
        multisets.sort()

        # dictの生成
        dict = {}
        acc += 1
        before = multisets[0].neighbors
        for multiset in multisets:
            if multiset.neighbors != before:
                acc += 1
            before = multiset.neighbors
            dict[tuple(multiset.neighbors)] = acc

        # label[h]の更新
        for multiset in multisets:
            newLabel = dict[tuple(multiset.neighbors)]
            graphs[multiset.graphIndex].l[h][multiset.v] = newLabel
            kernel[multiset.graphIndex, newLabel] += 1

        print("iteration: {} / {} ended!".format(h, H))
        yield kernel[:, np.any(kernel != 0, axis=0)]

if __name__ == '__main__':
    pass
