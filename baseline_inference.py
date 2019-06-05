import tensorflow as tf
from pymagnitude import *
import os
import csv
import logging
import numpy as np
import regex
import random

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

# hyperparameters
MAX_SEQ_LENGTH = 100
BATCH_SIZE = 16
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
    train_questions, train_answers, train_labels = read_QNLI_dataset(TRAINING_DATAFILE)
    test_questions, test_answers, test_labels = read_QNLI_dataset(DEV_DATAFILE)
    # batches per epoch
    num_batches_per_epoch_train = int(math.ceil(len(train_questions)/float(BATCH_SIZE)))
    num_batches_per_epoch_test = int(math.ceil(len(test_questions)/float(BATCH_SIZE)))
    
    # batchifying data
    def batchify(questions, answers, labels, batch_size=BATCH_SIZE):
        random.shuffle(questions)
        random.shuffle(answers)
        random.shuffle(labels)
        questions_batches = []
        answers_batches = []
        labels_batches = []
        index = 0
        iterations = int(math.floor(len(questions) / batch_size))
        for i in range(iterations):
            questions_batches.append(questions[index:index+BATCH_SIZE])
            answers_batches.append(answers[index:index+BATCH_SIZE])
            labels_batches.append(labels[index:index+BATCH_SIZE])
            index = index + BATCH_SIZE
        return questions_batches, answers_batches, labels_batches
    
    # generator for batches of training and testing data
    def batch_generator(questions, answers, labels):
        questions_batches, answers_batches, labels_batches = batchify(questions, answers, labels)
        while True:
            for question_batch, answer_batch, label_batch in zip(questions_batches, answers_batches, labels_batches):
                yield ([tf.keras.preprocessing.sequence.pad_sequences(vectors.query(question_batch), maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post', value=0),
                        tf.keras.preprocessing.sequence.pad_sequences(vectors.query(answer_batch), maxlen=MAX_SEQ_LENGTH, dtype='float32', padding='post', truncating='post', value=0)],
                        label_batch)
                        
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
    generator = batch_generator(train_questions, train_answers, train_labels),
    steps_per_epoch = num_batches_per_epoch_train,
    validation_data = batch_generator(test_questions, test_answers, test_labels),
    validation_steps = num_batches_per_epoch_test,
    epochs = EPOCHS,
)
    
    logger.debug("saving model to {MODEL_FILE}")
    model.save(MODEL_FILE)
    logger.debug("finished training")

train()
