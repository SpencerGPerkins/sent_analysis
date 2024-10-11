import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch.utils.data import TensorDataset, DataLoader
from nltk.corpus import stopwords
from collections import Counter
import string
import re
import seaborn as sns 
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


if is_cuda:
    device = torch.device("cuda")
    print("GPU is available.")
else:
    device = torch.device("cpu")
    print("No GPU, using CPU")

def preprocess_text(s):
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", '', s)
    # Replace all runs of whitespaces with no space
    s = re.sub(r"\s+", '', s)
    # replace digits with no space
    s = re.sub(r"\d", '', s)

    return s   

     
def main():
    df = pd.read_csv('../data/social_media/sentiment_analysis.csv')
    