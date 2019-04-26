
import pandas as pd
import numpy as np
from collections import Counter


def main():

    project = '/Users/deirdre/Documents/VA-ML/Project/EMS-Prediction/'    # path to main directory
    file_name = 'data/Watch-accuracy.csv'
    watch_data = pd.read_csv(project + file_name, header=0, sep=',', index_col=False)
    predictions = np.array(watch_data['pred'])
    true_clasess = np.array(watch_data['gt'])
    freq = {}

    # loop through predicted classes
    u_classes = np.unique(true_clasess)
    print(u_classes)
    for pc in u_classes:

        idx = np.where(predictions == pc)
        true_class = true_clasess[idx]
        freq[pc] = {}

        # find true classes dist
        for tc in u_classes:
            jdx = np.where(true_class == tc)
            freq[pc][tc] = len(jdx[0])

    print(freq)


main()
