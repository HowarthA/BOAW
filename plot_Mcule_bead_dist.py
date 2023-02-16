import pickle
import os
import numpy as np
from scipy.stats import gaussian_kde
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt
import os

folder = os.path.expanduser( "~/beads_dist/")

beads= [ ]

for f in sorted(os.listdir(folder)):

    if f.endswith(".p"):

        #b = pickle.load(open(folder + f, "rb"))

        pickle_ = pickle.load(open(folder + f, "rb"))

        for m in pickle_:

            if len(m) > 0:

                beads.extend(m)

        #b = [i for i in pickle.load(open(folder + f, "rb"))]

beads = np.array(beads)

print(np.shape(beads))

#pickle.dump(beads, open( folder +  "BOAW_Mcule_morfeus_beads.p","wb"))

scaler = StandardScaler()

scaler.fit(beads)

pickle.dump(scaler, open(folder + "BOAW_Mcule_morfeus_scaler.p","wb"))


quit()

scaler = pickle.load(open(folder + "BOAW_Mcule_morfeus_scaler.p","rb"))

beads = pickle.load( open(folder + "BOAW_Mcule_morfeus_beads.p","rb"))

beads = scaler.transform(beads)

pickle.dump(beads, open(folder + "BOAW_Mcule_morfeus_scaled_beads.p","wb"))

titles = ["abs charge", "masses", "logP", "MrC", "ASA", "TPSA", "Aromatic", "HBD", "HBA"]

boundaries = []

for c in range(np.shape(beads)[1]):

    sample = np.random.randint(0, len(beads) - 1, int(len(beads) * 0.1))

    print(len(sample))

    kde = gaussian_kde(beads[sample, c])

    print("made kde ", titles[c])

    min_ = np.min(beads[sample, c])

    max_ = np.max(beads[sample, c])

    x = np.linspace(min_, max_, 20)

    #

    y = kde.pdf(x)

    ddy = np.diff(y, 2)

    ddy1 = np.roll(ddy, 1)

    ddyn1 = np.roll(ddy, -1)

    w = np.where((ddy[1:-1] < ddy1[1:-1]) & (ddy[1:-1] < ddyn1[1:-1]))[0] + 2

    points = list(x[w])

    points = [min_] + points + [max_]

    boundaries.append(points)

    #

    pdf = kde.pdf(x)

    plt.plot(x, pdf, color="C" + str(c))

    for p in points:
        plt.axvline(p, linestyle="--", color="grey")

    plt.title(titles[c])

    plt.show()

    plt.close()

pickle.dump(boundaries, open(folder + "mcule_sample_morfeus_boundaries.p", "wb"))
