import tensorflow as tf
from pymagnitude import *
import os
import csv
from tqdm import tqdm
import logging
from collections import Counter
import pickle
import re
import regex
import logging
import numpy as np

# QNLI training data path
TRAINING_DATAFILE = "/Users/zxq001/QNLI/train.tsv"
# load pretrained embedding
vectors = Magnitude("/Users/zxq001/glove.840B.300d.magnitude")

# the maximimum length for the question sequence
MAX_SEQ_LENGTH = 100

# u = question sequence embedding (MAX_SEQ_LENGTH, 300) -> 1500D bidirectional LSTM -> maxpooling
q_in = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH, vectors.dim))
q_Bidir_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1500, activation='tanh', return_sequences=True), merge_mode='concat')(q_in)
expanded_q_LSTM = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(q_Bidir_LSTM)
q_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(MAX_SEQ_LENGTH, 1))(expanded_q_LSTM)
u =  tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(q_maxpool)

# v = answer sequence embedding (MAX_SEQ_LENGTH, 300) -> 1500D bidirectional LSTM -> maxpooling
a_in = tf.keras.layers.Input(shape=(MAX_SEQ_LENGTH, vectors.dim))
a_Bidir_LSTM = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(1500, activation='tanh', return_sequences=True), merge_mode='concat')(a_in)
expanded_a_LSTM = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x, axis=-1))(a_Bidir_LSTM)
a_maxpool = tf.keras.layers.MaxPooling2D(pool_size=(MAX_SEQ_LENGTH, 1))(expanded_a_LSTM)
v =  tf.keras.layers.Lambda(lambda x: tf.squeeze(x, axis=-1))(a_maxpool)