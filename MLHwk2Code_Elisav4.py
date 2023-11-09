# Marc Alessi 
# B Steele

"""
Created on Mon Oct  2 15:57:39 2023

@author: em
"""

"""
adapted from Jamin Rader's 
https://colab.research.google.com/github/eabarnes1010/course_ml_ats/blob/main/code/ann_ozone_joshuatree_filled.ipynb#scrollTo=oLyRWUOu4sMN

"""

#Hwk2, ATS 787A7
#Fully-connected or standard CNN to predict quantity  y, given inputs X
#as a classification task

#initial code copied from "Feed-forward neural network using ozone data at Joshua Tree"
#by Jamin Rader at CSU

#////////0. Set Up Environments////////

import sys
import numpy as np
import seaborn as sb

import pandas as pd
import datetime
import tensorflow as tf

# import tensorflow.keras as keras
import sklearn

# import pydot
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# %matplotlib inline

# tf.compat.v1.disable_v2_behavior()

print(f"python version = {sys.version}")
print(f"numpy version = {np.__version__}")
print(f"tensorflow version = {tf.__version__}")

Path = '/Users/em/Documents/Courses 2022-2024/ATS 780A7_Machine_Learning/Hwk2/'

#////////1. Data Preparation////////

#1.1 Data Overview
# x values: day month year date_time PSD_sum PSD_max PSD_min
# y values: RSSI_ext modified to classes 0, 1, and 2
# classification (0, 1, or 2 as low, medium, and high)
# standardization (0 to 1 values for each x)

#1.2 Define Input and Output
# inputs (x): day month year date_time PSD_sum PSD_max PSD_min
# output (y): RSSI_ext modified to classes 0, 1, and 2

#1.3 Visualizing our Data
# spectrograms plotted in Matlab
# used excel, plotted distribution of RSSI_ext

#1.4 Partitioning Data in Training, Validation, and Testing Sets
# Jan 1 2015 to 31 Dec 2016; Apr to Oct are quiet; Nov to Mar are loud.
# 1 Nov 2014 to 1 Nov 2015 is a full set of loud to quiet, one year
# 2 Nov 2015 to 31 Dec 2016 is a full set plus 2 months of loud. 

# Import training, validation and test data from csv
Xtrain = pd.read_csv('Xtrain_data_inputs3.csv', on_bad_lines='skip')# the the inputs (X)
Xtrain = Xtrain.to_numpy()

Ttrain = pd.read_csv('Ttrain_data_outputs.csv', on_bad_lines='skip')# the outputs (T is for target)
Ttrain = Ttrain.to_numpy()

Xval = pd.read_csv('Xval_data_inputs3.csv', on_bad_lines='skip')
Xval = Xval.to_numpy()

Tval = pd.read_csv('Tval_data_outputs.csv', on_bad_lines='skip')
Tval = Tval.to_numpy()

Xtest = pd.read_csv('Xtest_data_inputs3.csv', on_bad_lines='skip')
Xtest = Xtest.to_numpy()

Ttest = pd.read_csv('Ttest_data_outputs.csv', on_bad_lines='skip') #not sure when to load test solutions
Ttest = Ttest.to_numpy()

#x_train = pd.read_csv('Xtrain_data_inputs2.csv', on_bad_lines='skip')# the the inputs (X)
#y_train = pd.read_csv('Ttrain_data_outputs.csv', on_bad_lines='skip')# the outputs (T is for target)
#x_val = pd.read_csv('Xval_data_inputs2.csv', on_bad_lines='skip')
#y_val = pd.read_csv('Tval_data_outputs.csv', on_bad_lines='skip')
#x_test = pd.read_csv('Xtest_data_inputs2.csv', on_bad_lines='skip')
#y_test = pd.read_csv('Ttest_data_outputs.csv', on_bad_lines='skip') #not sure when to load test solutions


#/////////BUILD MODEL ///////////

def build_model(x_train, y_train, settings):
    # create input layer
    input_layer = tf.keras.layers.Input(shape=x_train.shape[1:])

    # create a normalization layer if you would like
    normalizer = tf.keras.layers.Normalization(axis=(1,))
    normalizer.adapt(x_train)
    layers = normalizer(input_layer)

    # create hidden layers each with specific number of nodes
    assert len(settings["hiddens"]) == len(
        settings["activations"]
    ), "hiddens and activations settings must be the same length."

    # add dropout layer, dropout nodes stop overfitting. For example, "0.3" drops out 30% of nodes. 
    layers = tf.keras.layers.Dropout(rate=settings["dropout_rate"])(layers)

    # create output layer
    output_layer = tf.keras.layers.Dense(
        units=y_train.shape[-1],
        activation="softmax",
        use_bias=True,
        bias_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 1),
        kernel_initializer=tf.keras.initializers.RandomNormal(seed=settings["random_seed"] + 2),
    )(layers)

    # construct the model
    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    model.summary()

    return model


def compile_model(model, settings):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=settings["learning_rate"]),
        loss=tf.keras.losses.CategoricalCrossentropy(),
        metrics=[
            tf.keras.metrics.CategoricalAccuracy(),
        ],
    )
    return model

settings = {
    "hiddens": [2, 2],
    "activations": ["relu", "relu"],
    "learning_rate": 0.001,
    "random_seed": 33,
    "max_epochs": 1_000,
    "batch_size": 256,
    "patience": 10,
    "dropout_rate": 0.3,
}

tf.keras.backend.clear_session()
tf.keras.utils.set_random_seed(settings["random_seed"])

model = build_model(Xtrain, Ttrain, settings)
model = compile_model(model, settings)

# model = build_model(x_train, y_train, settings)
# model = compile_model(model, settings)

#////////// training the model ///////////


# define the class weights; this is where you can change the 1's at the end to manually tune the weight for each class
# if you have a low number of samples classified as class 1, for example, you can weight that class higher so that 
# the system doesn't dismiss class 1 for lack of training data and rarely output class 1.
# class_weights = {
#     0: 1 / np.mean(Ttrain[:, 0] == 1),
#     1: 1 / np.mean(Ttrain[:, 1] == 1),
#     2: 1 / np.mean(Ttrain[:, 2] == 1),
# }
# class_weights = {0: 1, 1: 1, 2: 1}

# define the early stopping callback
early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss", patience=settings["patience"], restore_best_weights=True, mode="auto"
)

# train the model via model.fit
history = model.fit(
    Xtrain,
    Ttrain,
    epochs=settings["max_epochs"],
    batch_size=settings["batch_size"],
    shuffle=True,
    validation_data=[Xval, Tval],
    #class_weight=class_weights,
    callbacks=[early_stopping_callback],
    verbose=1,
)

#///

# Let's plot the change in loss and categorical_accuracy

fig, axs = plt.subplots(1, 2, figsize=(10, 4))

axs[0].plot(history.history["loss"], label="training")
axs[0].plot(history.history["val_loss"], label="validation")
axs[0].set_xlabel("Epoch")
axs[0].set_ylabel("Loss")
axs[0].legend()

axs[1].plot(history.history["categorical_accuracy"], label="training")
axs[1].plot(history.history["val_categorical_accuracy"], label="validation")
axs[1].set_xlabel("Epoch")
axs[1].set_ylabel("Categorical Accuracy")
axs[1].legend()

#///

# What predictions did the model make for our training, validation, and test sets?
Ptrain = model.predict(Xtrain)  # Array of class likelihoods for each class
Pval = model.predict(Xval)
Ptest = model.predict(Xtest)

Cptrain = Ptrain.argmax(axis=1)  # 1-D array of predicted class (highest likelihood)
Cpval = Pval.argmax(axis=1)
Cptest = Ptest.argmax(axis=1)

Cttrain = Ttrain.argmax(axis=1)  # 1-D array of truth class
Ctval = Tval.argmax(axis=1)
Cttest = Ttest.argmax(axis=1)


##///////

from sklearn.metrics import f1_score, accuracy_score

print("Validation Categorical Accuracy:", accuracy_score(Ctval, Cpval))

# Weight equal to the inverse of the frequency of the class
cat_weights = np.sum((1 / np.mean(Ttrain, axis=0)) * Tval, axis=1)
print("Validation Weighted Categorical Accuracy:", accuracy_score(Ctval, Cpval, sample_weight=cat_weights))

#//// STOP HERE, manually create confusion matrix
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#notes from Libby: 

#true pos count
#true neg count
#false pos count
#false neg coount
#sk learn metrics confusion matrix
#list of true classes as input
#list of predicted classes as input
#model.predict
#argmax

#y_pred = model.predictIinput)
# matrix number of sampmles in y axis, 
#y_class = np.argmax(y_predict) argmax tells you which col has max val



# def confusion_matrix(predclasses, targclasses):
#     class_names = np.unique(targclasses)

#     table = []
#     for pred_class in class_names:
#         row = []
#         for true_class in class_names:
#             row.append(100 * np.mean(predclasses[targclasses == true_class] == pred_class))
#         table.append(row)
#     class_titles_t = ["T(Good)", "T(Fair)", "T(Poor)"]
#     class_titles_p = ["P(Good)", "P(Fair)", "P(Poor)"]
#     conf_matrix = pd.DataFrame(table, index=class_titles_p, columns=class_titles_t)
#     display(conf_matrix.style.background_gradient(cmap="Blues").format("{:.1f}"))
    
#////
    
# print("Predicted versus Target Classes")
# confusion_matrix(Cptrain, Cttrain)
# confusion_matrix(Cpval, Ctval)
# confusion_matrix(Cptest, Cttest)

# df_class0 = pd.DataFrame(O3val[Cpval == 0])
# df_class1 = pd.DataFrame(O3val[Cpval == 1])
# df_class2 = pd.DataFrame(O3val[Cpval == 2])

# sb.violinplot(data=[O3val[Cpval == 0], O3val[Cpval == 1], O3val[Cpval == 2]])

# plt.axhline(55, color="goldenrod", zorder=0)
# plt.axhline(71, color="red", zorder=0)
# plt.ylabel("Ozone (ppb)")
# plt.xlabel("Predicted Class")
# plt.xticks(
#     [0, 1, 2],
#     ["Good", "Fair", "Poor"],
# )
# plt.title("O3 Concentrations when each Class is Predicted", fontweight="demi")
# plt.show()

#///

