import numpy as np
import random

class Graph:
    def __init__(self, n=1000, c=0, H=100):
        self.n = n  # number of vertexes
        self.A = np.zeros((self.n, self.n), dtype=np.bool)  # adjancy matrix
        self.l = np.full((H, n), 0, dtype=np.int)  # label
        self.c = c  # class

    def addEdge(self, i, j):
        assert i != j
        self.A[i, j] = True
        self.A[j, i] = True

    def setLabelWithDegree(self):
        for i in range(self.n):
            self.l[0][i] = 0
            for j in range(self.n):
                if self.A[i, j]:
                    self.l[0][i] += 1

    def setLabelWithReachable(self, x):
        reachables = np.zeros((self.n, self.n), dtype=np.int)
        visited = [False for _ in range(self.n)]
        def dfs(start, now, xx):
            if xx == 0:
                return
            visited[now] = True
            for i in range(self.n):
                if self.A[now, i] and (not visited[i]):
                    reachables[start][i] += 1
                    dfs(start, i, xx-1)
            visited[now] = False
        for i in range(self.n):
            dfs(i, i, x)
        print(reachables)
        reachable_counts = np.array([np.sum(reachable) for reachable in reachables])
        print(reachable_counts)
                    
    def adjacencyList(self, i):
        return np.arange(self.A[i].size)[self.A[i]]

class RandomGraph(Graph):
    def __init__(self, n=1000, c=0, p=0.5):
        super().__init__(n=n, c=c)
        for i in range(n):
            for j in range(i + 1, n):
                if random.random() < p:
                    self.addEdge(i, j)

class PlantedClique(RandomGraph):
    def __init__(self, n=1000, c=1, p=0.5, beta=1.6):
        super().__init__(n=n, c=c, p=p)
        k = int(np.ceil(beta * np.sqrt(n)))
        for i in range(k):
            for j in range(i + 1, k):
                self.addEdge(i, j)
