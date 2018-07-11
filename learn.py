from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import WL_kernel
import graph

def learn(num, H, setLabel, beta=1.6):
    graphs = []
    for i in range(num):
        randomGraph = graph.RandomGraph(n=100)
        if setLabel:
            randomGraph.setLabelWithDegree()
        graphs.append(randomGraph)
        plantedClique = graph.PlantedClique(n=100, beta=beta)
        if setLabel:
            plantedClique.setLabelWithDegree()
        graphs.append(plantedClique)
    for kernel in WL_kernel.WLKernel(graphs, H):
        classes = [g.c for g in graphs]
        X_learn = kernel[:num]
        y_learn = classes[:num]
        X_test = kernel[num:]
        y_test = classes[num:]
            
        clf = LinearSVC(random_state=0)
        clf.fit(X_learn, y_learn)
        predictions = clf.predict(X_test)
        
        correct = 0
        for prediction, ans in zip(predictions, y_test):
            if prediction == ans:
                correct += 1
        print("{} / {} ({} %)".format(correct, num, correct * 100 / num))
        yield correct / num

if __name__ == '__main__':
    betas = [1.8, 1.4, 1.0, 0.6]
    H_MAX = 5
    Hs = range(H_MAX)
    for beta in betas:
        y_nodeDegree = list(learn(500, H_MAX, True, beta=beta))
        y_none = list(learn(500, H_MAX, False, beta=beta))
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.ylim([0, 1])
        plt.plot(Hs, y_nodeDegree, marker="o", label="Degree")
        plt.plot(Hs, y_none, marker="o", label="None")
        plt.legend()
        plt.title("beta = {}".format(beta))
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.show()
