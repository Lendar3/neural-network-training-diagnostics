import csv
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


X_list = []
Y_list = []

with open( PROJECT_ROOT / "data" / "raw" / "HIGGS.csv", mode='r' ,encoding='utf-8', newline='') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        Y_list.append([float(row[0])])
        X_list.append([float(val) for val in row[1:29]])

X = np.array(X_list, dtype=np.float32)
Y = np.array(Y_list, dtype=np.float32)
np.savez(
    PROJECT_ROOT / "data" / "processed" / "higgs.npz",
    X=X,
    Y=Y,)