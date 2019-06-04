import tensorflow as tf
from pymagnitude import *
import os
import csv
import logging
import numpy as np
import regex

# QNLI training and test data path
TRAINING_DATAFILE = "/home/ec2-user/QNLI/train.tsv"
DEV_DATAFILE = "/home/ec2-user/QNLI/dev.tsv"
# load pretrained embedding
vectors = Magnitude("/home/ec2-user/glove.840B.300d.magnitude")
MODEL_FILE = "/home/ec2-user/model.h5"

# prepare logger
logger = logging.getLogger(__name__)
logger.addHandler(logging.StreamHandler())
logger.setLevel(logging.DEBUG)
tf.logging.set_verbosity(logging.ERROR)

# the maximimum length for the question sequence
MAX_SEQ_LENGTH = 100
BATCH_SIZE = 64
EPOCHS = 100

# read QNLI dataset from GLUE benchmark
def read_QNLI_dataset(FILE_PATH):
    trainfile = open(FILE_PATH)
    trainfile = csv.reader(trainfile, delimiter='\t')
    questions_raw = []
    answers_raw = []
    labels = []
    # skip the header
    next(trainfile)
    for row in trainfile:
        questions_raw.append(row[1])
        answers_raw.append(row[2])
        if row[3] == "entailment":
            labels.append(1)
        else:
            labels.append(0)
    # seperate the questions and answers into words
    questions = []
    answers = []
    for question in questions_raw:
        questions.append(regex.findall(r"[^[:punct:] ]+|[[:punct:]]", question))
    for answer in answers_raw:
        answers.append(regex.findall(r"[^[:punct:] ]+|[[:punct:]]", answer))
    return questions, answers, labels

# trains and saves the model
def train():
    logger.info("training")
    
    logger.debug("loading data")
    # load data
    questions, answers, labels = read_QNLI_dataset(TRAINING_DATAFILE)
    
    logger.debug("preprocessing data")
    # batchify training and testing data
    train_questions, train_answers, train_labels = read_QNLI_dataset(TRAINING_DATAFILE)
    test_questions, test_answers, test_labels = read_QNLI_dataset(DEV_DATAFILE)
    training_batches = MagnitudeUtils.batchify([train_questions, train_answers], train_labels, BATCH_SIZE)
    testing_batches = MagnitudeUtils.batchify([test_questions, test_answers], test_labels, BATCH_SIZE)

    num_batches_per_epoch_train = int(math.ceil(len(train_questions)/float(BATCH_SIZE)))
    num_batches_per_epoch_test = int(math.ceil(len(test_questions)/float(BATCH_SIZE)))

    # batch generator
    # Generates batches of the transformed training data
    train_batch_generator = (
      (
        vectors.query(X_train_batch), # Magnitude will handle converting the 2D array of text into the 3D word vector representations!
        MagnitudeUtils.to_categorical(y_train_batch, num_outputs) # Magnitude will handle converting the class labels into one-hot encodings!
      ) for X_train_batch, y_train_batch in training_batches
    )

    logger.debug("building model")
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

    # define the concatenation function for the Lambda layer
    def concat_u_v(uv):
        u = uv[0]
        v = uv[1]
        return tf.concat([u, v, tf.math.abs(u-v), tf.math.multiply(u, v)], -1)

    # output = concatenation layer (u, v, |u-v|, u*v) -> 512D hidden layer -> output node
    concat_output = tf.keras.layers.Lambda(concat_u_v)([u, v])
    hidden = tf.keras.layers.Dense(512)(concat_output)
    output = tf.keras.layers.Dense(1, activation="softmax")(hidden)
    model = tf.keras.Model(inputs=[q_in, a_in], outputs=output)
    model.summary()
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    
    logger.debug("fitting model")
    model.fit_generator(
    generator = train_batch_generator,
    steps_per_epoch = num_batches_per_epoch_train,
    validation_data = test_batch_generator,
    validation_steps = num_batches_per_epoch_test,
    epochs = EPOCHS,
)
    
    logger.debug("saving model to {MODEL_FILE}")
    model.save(MODEL.FILE)
    logger.debug("finished training")

train()
