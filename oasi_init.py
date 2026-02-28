# coding: utf-8

import os
import json
import csv
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import pickle
from itertools import cycle
from functools import partial

# Scikit-learn
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn import linear_model
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report, roc_auc_score, roc_curve

# TensorFlow / Keras
import tensorflow as tf
import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, SeparableConv2D, GlobalAveragePooling2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.mixed_precision import Policy, set_global_policy

# ==========================================
# 1. Environment & GPU Setup
# ==========================================
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configuration successful. Num GPUs Available: ", len(gpus))
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected.")

# Set mixed precision policy
policy = Policy('mixed_float16')
set_global_policy(policy)

# ==========================================
# 2. Data Loading & Preparation
# ==========================================
def load_data(data_path):
    """Loads training, testing, and validation dataset from .npy files."""
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_validation = np.load(f'{data_path}/X_val.npy')
 
    y_train = np.load(f'{data_path}/y_train.npy')[..., np.newaxis]
    y_test = np.load(f'{data_path}/y_test.npy')[..., np.newaxis]
    y_validation = np.load(f'{data_path}/y_val.npy')[..., np.newaxis]
    
    print("Dataset loaded!")
    return X_train, X_test, X_validation, y_train, y_test, y_validation

def prepare_dataset(data_path):
    """Scales and reshapes datasets."""
    X_train, X_test, X_validation, y_train, y_test, y_validation = load_data(data_path)
    
    scaler = StandardScaler()
    
    # Scale Train
    num_instances, num_time_steps, num_features = X_train.shape
    X_train = scaler.fit_transform(X_train.reshape(-1, num_features)).reshape(num_instances, num_time_steps, num_features)
    
    # Scale Test
    num_instances, num_time_steps, num_features = X_test.shape
    X_test = scaler.fit_transform(X_test.reshape(-1, num_features)).reshape(num_instances, num_time_steps, num_features)
    
    # Scale Validation
    num_instances, num_time_steps, num_features = X_validation.shape
    X_validation = scaler.fit_transform(X_validation.reshape(-1, num_features)).reshape(num_instances, num_time_steps, num_features) 
    
    # Add axis for CNN input
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]
    X_validation = X_validation[..., np.newaxis]

    return X_train, y_train, X_validation, y_validation, X_test, y_test

# ==========================================
# 3. Hyperparameters & Configuration
# ==========================================
DATA_PATH = "/home/22EC1102/soumen/data/kws_10_log_mel"
class_names = ['off', 'left', 'down', 'up', 'go', 'on', 'stop', 'unknown', 'right', 'yes']
EPOCHS = 30 #100
BATCH_SIZE = 64 #16
PATIENCE = 5
LEARNING_RATE = 0.0001
CLASS = 10

X_train, y_train, X_validation, y_validation, X_test, y_test = prepare_dataset(DATA_PATH)
input_shape = (X_train.shape[1], X_train.shape[2], 1)

# ==========================================
# 4. Objective Functions & Perturbation
# ==========================================
def objective_accuracy(model, X_val, y_val):
    """Compute validation accuracy."""
    _, accuracy = model.evaluate(X_val, y_val, verbose=0)
    return accuracy

def objective_model_size(model):
    """Compute model size via parameter count in MB."""
    # count_params() safely handles all variable types under the hood
    return model.count_params() * 4 / (1024 ** 2)

def perturb_solution(solution, bounds):
    """Generate a perturbed solution within the given bounds."""
    return [
        max(bounds[0][0], min(bounds[0][1], solution[0] + random.randint(-1, 1))),
        max(bounds[1][0], min(bounds[1][1], solution[1] + random.randint(-32, 32))),
        random.choice(bounds[2]),
        # [max(bounds[4][0], min(bounds[4][1], fc + random.randint(-64, 64))) for fc in solution[3]],
        max(bounds[3][0], min(bounds[3][1], solution[3] + random.randint(-1, 1))),
        random.choice(bounds[4]),
        random.choice(bounds[5])
    ]

# ==========================================
# 5. Model Architecture (DSCNN)
# ==========================================

def create_dscnn_model(conv_layers, filters, kernel_size, fc_layers, use_bn, use_dropout,input_shape, depth_multiplier=1):
    input_shape = (X_train.shape[1], X_train.shape[2], 1)
    model = Sequential()
    
    # First standard convolutions
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu', input_shape=input_shape))
    model.add(Conv2D(filters, kernel_size=kernel_size, strides=(1, 1), padding='same', activation='relu'))
    
    if model.output_shape[1] > 2 and model.output_shape[2] > 2:
        model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))
    
    # Depthwise Separable Blocks
    def depthwise_separable_block(f_out):
        model.add(SeparableConv2D(f_out, (3, 3), depth_multiplier=depth_multiplier, padding="same", activation="relu"))
        if use_bn: model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(SeparableConv2D(f_out, (3, 3), depth_multiplier=depth_multiplier, padding="same", activation="relu"))
        if use_bn: model.add(BatchNormalization())
        model.add(Activation('relu'))
            
        if model.output_shape[1] > 2 and model.output_shape[2] > 2:
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Dynamic scaling based on layers
    curr_filters = filters
    for _ in range(conv_layers):
        depthwise_separable_block(curr_filters)
        curr_filters = min(curr_filters * 2, 512)

    # model.add(Flatten())
    model.add(GlobalAveragePooling2D())

    # Fully connected block (Mapping int to architecture depth)
    if fc_layers == 4:
        model.add(Dense(512, activation='relu'))
        if use_dropout: model.add(Dropout(0.5))
    elif fc_layers == 3:
        model.add(Dense(256, activation='relu'))
        if use_dropout: model.add(Dropout(0.5))
    elif fc_layers == 2:
        model.add(Dense(128, activation='relu'))
        if use_dropout: model.add(Dropout(0.3))
    elif fc_layers == 1:
        model.add(Dense(64, activation='relu'))
        if use_dropout: model.add(Dropout(0.3))

    model.add(Dense(CLASS, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

# ==========================================
# 6. Simulated Annealing Optimization
# ==========================================
def chaos_simulated_annealing(bounds, max_iter, cooling_rate=0.012, initial_temps=(10, 10), input_shape=input_shape, T_min=1e-6):
    """Runs Chaos Simulated Annealing multi-objective optimization."""
    current_solution = [
        random.randint(*bounds[0]),                                        # Conv layers
        random.randint(*bounds[1]),                                        # Filters
        random.choice(bounds[2]),                                          # Kernel size
        # [random.randint(*bounds[4]) for _ in range(random.randint(*bounds[3]))],  # FC layers
        random.randint(*bounds[3]),
        random.choice(bounds[4]),                                          # Batch Normalization
        random.choice(bounds[5])                                           # Dropout
    ]
    temperatures = list(initial_temps)
    archive_all = []
    eta = 0.01

    for iteration in range(max_iter):
        print(f"Iteration {iteration+1}/{max_iter}")
        
        if all(t < T_min for t in temperatures):
            print("Temperatures are below the minimum threshold. Stopping.")
            break
            
        new_solution = perturb_solution(current_solution, bounds)
        
        # Build and evaluate models for current and new solutions
        current_model = create_dscnn_model(*current_solution, input_shape=input_shape)
        new_model = create_dscnn_model(*new_solution, input_shape=input_shape)
        
        early_stopping = EarlyStopping(monitor='loss', patience=PATIENCE)
        current_model.fit(X_train, y_train, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)
        new_model.fit(X_train, y_train, epochs=200, batch_size=32, callbacks=[early_stopping], verbose=0)

        current_obj = [
            objective_accuracy(current_model, X_validation, y_validation),
            objective_model_size(current_model)
        ]
        new_obj = [
            objective_accuracy(new_model, X_validation, y_validation),
            objective_model_size(new_model)
        ]
        
        print(f"current_obj: {current_obj}")
        print(f"new_obj: {new_obj}")
        
        # Acceptance probabilities
        accept_probs = [
            1 if new_obj[0] > current_obj[0] else np.exp(-(current_obj[0] - new_obj[0]) / temperatures[0]),
            1 if new_obj[1] < current_obj[1] else np.exp(-(new_obj[1] - current_obj[1]) / temperatures[1])
        ]

        # Check acceptance
        all_accepted = True
        if any(p == 1 for p in accept_probs):
            current_solution = new_solution
            archive_all.append((new_solution, new_obj))
        else:
            for p in accept_probs:
                if random.random() >= p:
                    all_accepted = False
                    break
            if all_accepted:
                current_solution = new_solution
                archive_all.append((new_solution, new_obj))    

        accuracy_improvement = (new_obj[0] - current_obj[0]) / (current_obj[0] + 1e-6)
        memory_improvement = (current_obj[1] - new_obj[1]) / (current_obj[1] + 1e-6)

        # Exponential cooling
        temperatures = [
            t * (1 + eta * accuracy_improvement) if i == 0 else t * (1 + eta * memory_improvement)
            for i, t in enumerate(temperatures)
        ]
        
        print("###############################################")
    
    return archive_all

# ==========================================
# 7. Execution & Storage
# ==========================================
bounds = [
    (1, 4),                     # Number of convolutional layers (Adjusted bound slightly for DSCNN scaling)
    (16, 64),                   # Number of base filters
    [(3, 3), (5, 5)],           # Kernel size
    (1, 3),                     # Number of fully connected layers
    # (128, 512),                 # Number of neurons per fully connected layer
    [True, False],              # Batch Normalization
    [True, False]               # Dropout
]

def store_pareto_archive_all(archive_all, filename="OASI_init.xlsx"):
    """Stores the final Pareto archive in an Excel file."""
    data = []
    for idx, (solution, objectives) in enumerate(archive_all):
        record = {
            "Solution ID": f"Solution_{idx + 1}",
            "Conv Layers": solution[0],
            "Filters": solution[1],
            "Kernel Size": str(solution[2]),
            # "FC Layers": "-".join(map(str, solution[3])),
            "FC Layers": solution[3],
            "Batch Normalization": solution[4],
            "Dropout": solution[5],
            "Accuracy": objectives[0],
            "Model Size_MB": objectives[1]
        }
        data.append(record)
    
    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Pareto archive_all saved to {filename}")

if __name__ == "__main__":
    archive_all = chaos_simulated_annealing(bounds, max_iter=30)            
    store_pareto_archive_all(archive_all)