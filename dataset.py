# dataset.py

import numpy as np
import os
import pathlib

sequence_len = 700
original_features = 57
total_features = 42
amino_acid_residues = 21
num_classes = 8

def get_dataset(path):
    """
    returns correct dataset format where ret contains (# of samples, sequence_length of sample, input features (one-hot encoded AA + PSSM features) + one-hot for secondary structure (actual class)

    Each amino acid in the sequence encoded as a vector where first 21 residues 
    """
    ds = np.load(path)
    ds = ds.reshape((ds.shape[0], sequence_len, original_features))

    ret = np.zeros((ds.shape[0], ds.shape[1], total_features + num_classes))
    ret[:, :, 0:amino_acid_residues] = ds[:, :, 0:amino_acid_residues] #get one-hot encodings for AA in first 21 positions for each sample
    ret[:, :, amino_acid_residues:total_features] = ds[:, :, 35:56] #get PSSM features in next 21 positions for each sample/input
    ret[:, :, total_features:total_features+num_classes] = ds[:, :, 22:30] #get secondary structure classification 
    
    return ret

def split_with_shuffle(dataset, seed=None):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    n = dataset.shape[0]
    return (dataset[:int(n*0.8)], dataset[int(n*0.8):int(n*0.9)], dataset[int(n*0.9):])
