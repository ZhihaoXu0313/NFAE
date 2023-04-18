import os
import re
import yaml
import numpy as np
import matplotlib.pyplot as plt


def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)


def grep(target, file):
    results = []
    with open(file, 'r') as f:
        for line in f:
            if re.search(target, line):
                results.append(line.rstrip('\n'))
    return results


def plot_band(band, band_label):
    with open("band.yaml", 'r') as stream:
        data = yaml.safe_load(stream)
    phon = data['phonon']
    Natom = data['natom']
    print('Number of atoms:', Natom)
    Nkp = len(phon)
    print('Number of k-points:', Nkp)

    lines = []
    for i in range(Nkp):
        distance = phon[i]['distance']
        band = phon[i]['band']
        Nband = len(band)
        lines.append("%8.5f\t" % distance)
        for j in range(Nband):
            eig = band[j]['frequency']
            lines.append("%8.5f\t" % eig)

        lines.append("\n")

    print('Number of bands:', Nband)
    print("Begin write band.txt ...")
    with open("band.txt", 'w') as f:
        f.writelines(lines)

    bandtxt = np.loadtxt("band.txt", dtype=float)
    kpts = bandtxt[:, 0]

    split_band = []
    temp_list = []
    for i in band:
        if i == ",":
            split_band.append(temp_list)
            temp_list = []
        else:
            temp_list.append(i)
    split_band.append(temp_list)

    split_label = []
    n_subplt = len(split_band)
    i = 0
    for n in range(n_subplt):
        split_label.append(band_label[i: i + len(split_band[n])])
        i = i + len(split_band[n])

    start = 0
    for n in range(n_subplt):
        plt.subplot(1, n_subplt, n + 1)
        if n != 0:
            plt.tick_params(left=False, right=False, labelleft=False)
        else:
            plt.ylabel("Frequency (THz)")
        end = start + 51 * (len(split_band[n]) - 1)
        for k in range(Nband):
            plt.plot(kpts[start:end], bandtxt[start:end, k + 1], color='r', linestyle='-', linewidth=1)
            plt.axhline(y=0.0, color='b', linestyle='-', linewidth=1.3)
            plt.xlim([kpts[start], kpts[end-1]])
            plt.xticks(np.append(kpts[start:end:51], kpts[end-1]), split_label[n])
        start = end

    plt.tight_layout()
    plt.show()
    plt.savefig("band.png")

    print("Done!!")
