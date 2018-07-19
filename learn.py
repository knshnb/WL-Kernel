import copy
import pickle
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import WL_kernel
import graph

def learn(graphs, H):
    for kernel in WL_kernel.WLKernel(graphs, H):
        classes = [g.c for g in graphs]
        X_learn = kernel[:len(graphs)//2]
        y_learn = classes[:len(graphs)//2]
        X_test = kernel[len(graphs)//2:]
        y_test = classes[len(graphs)//2:]
            
        clf = LinearSVC(random_state=0)
        clf.fit(X_learn, y_learn)
        predictions = clf.predict(X_test)
        
        correct = 0
        for prediction, ans in zip(predictions, y_test):
            if prediction == ans:
                correct += 1
        print("{} / {} ({} %)".format(correct, len(graphs)//2, correct * 100 / (len(graphs)//2)))
        yield correct / (len(graphs)//2)

if __name__ == '__main__':
    # betas = [1.8, 1.4, 1.0, 0.6]
    betas = [0.7, 0.8, 0.9]
    H_MAX = 5
    Hs = range(H_MAX)
    NUM = 10

    for beta in betas:
        with open('beta{}_num_500.pickle'.format(beta), 'rb') as f:
            graphs = pickle.load(f)

        graphs_nodeDegree = copy.deepcopy(graphs)
        for g in graphs_nodeDegree:
            g.setLabelWithDegree()

        y_nodeDegree = list(learn(graphs_nodeDegree, H_MAX))
        y_none = list(learn(graphs, H_MAX))
        plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
        plt.ylim([0, 1])
        plt.plot(Hs, y_nodeDegree, marker="o", label="Degree")
        plt.plot(Hs, y_none, marker="o", label="None")
        plt.legend()
        plt.title("beta = {}".format(beta))
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.show()
