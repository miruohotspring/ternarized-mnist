from pathlib import Path
import requests
import pickle
import gzip
from matplotlib import pyplot
import numpy as np
import codecs
import os

AMOUNT = 100
DATA_PATH = Path("data")
PATH = DATA_PATH / "mnist"

PATH.mkdir(parents=True, exist_ok=True)

URL = "http://deeplearning.net/data/mnist/"
FILENAME = "mnist.pkl.gz"

if not (PATH / FILENAME).exists():
        content = requests.get(URL + FILENAME).content
        (PATH / FILENAME).open("wb").write(content)
#print("download done")

with gzip.open((PATH / FILENAME).as_posix(), "rb") as f:
        ((x_train, y_train), (x_valid, y_valid), _) = pickle.load(f, encoding="latin-1")

#pyplot.imshow(x_valid[0].reshape((28, 28)), cmap="gray")
#pyplot.show()


left = f"float x_test[{AMOUNT}][28][28] = [ \\\n"
print(left, file=codecs.open('data.c', 'a', 'utf-8'))
for i in range(AMOUNT):
    right = np.array2string(x_valid[i].reshape((28, 28)), separator=',') + ","
    right = right.replace('[', '{').replace(']', '}')
    print(right, file=codecs.open('data.c', 'a', 'utf-8'))
print("};", file=codecs.open('data.c', 'a', 'utf-8'))

left = f"float y_test[{AMOUNT}] = [ \\\n"
left = left.replace('[', '{').replace(']', '}')
print(left, file=codecs.open('data.c', 'a', 'utf-8'))
for i in range(AMOUNT):
    right = str(y_valid[i]) + ", "
    print(right, file=codecs.open('data.c', 'a', 'utf-8'))
print("};", file=codecs.open('data.c', 'a', 'utf-8'))
