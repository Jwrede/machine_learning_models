from scipy.io import loadmat
import numpy as np
from machine_learning_models import multi_class_classifier
from IPython import get_ipython
import matplotlib.pyplot as plt
def __reset__(): get_ipython().magic('reset -sf')

mat = loadmat(r"C:\Users\Jonathan\Desktop\coursera\Machine learning\Machine learning models Python\models\nn_data\ex4data1.mat")
features = mat["X"]
labels = mat["y"]

