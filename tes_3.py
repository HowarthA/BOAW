import pickle
import  os
import numpy as np
from scipy.stats import gaussian_kde
from matplotlib import  pyplot as plt


beads = []

folder = "/Users/alexanderhowarth/Desktop/beads_dist/"

for f in sorted(os.listdir(folder)):

    if f.endswith(".p"):

        b = [i for i in pickle.load(open(folder + f,"rb"))]

        if len(b)>9:

            beads.extend(b)


beads = np.array(beads)

titles = ["abs charge", "masses", "logP", "MrC", "ASA", "TPSA", "Aromatic", "HBD", "HBA"]

boundaries = []

for c in range(np.shape(beads)[1]):

    sample = np.random.randint(0,len(beads)-1,int(len(beads)*0.1))

    print(len(sample))

    kde = gaussian_kde(beads[sample,c])

    print("made kde ", titles[c])

    min_ = np.min(beads[sample,c])

    max_ = np.max(beads[sample,c])

    x = np.linspace(min_,max_,20)

    #

    y = kde.pdf(x)

    ddy = np.diff(y, 2)

    ddy1 = np.roll(ddy, 1)

    ddyn1 = np.roll(ddy, -1)

    w = np.where((ddy[1:-1] < ddy1[1:-1]) & (ddy[1:-1] < ddyn1[1:-1]))[0] + 2

    points = list(x[w])

    points = [min_] + points +[max_]

    boundaries.append(points)

    #

    pdf = kde.pdf(x)

    plt.plot(x, pdf, color="C" + str(c))

    for p in points:
        plt.axvline(p, linestyle="--", color="grey")

    plt.title(titles[c])

    plt.show()

    plt.close()
    






pickle.dump(boundaries,open("mcule_sample_boundaries.p","wb"))
