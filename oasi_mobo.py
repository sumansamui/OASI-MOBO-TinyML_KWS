#!/usr/bin/env python
# coding: utf-8

import os
import time
import json
import itertools
import random
import ast
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Scipy for QMC Initialization (LHS, Sobol)
from scipy.stats import qmc

# Scikit-learn imports
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# TensorFlow / Keras
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Flatten, Dense, Dropout, SeparableConv2D
from tensorflow.keras.callbacks import EarlyStopping

# Pymoo
from pymoo.indicators.hv import HV
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

# ==========================================
# 1. GPU Configuration
# ==========================================
try:
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print("GPU configuration successful.")
    else:
        print("No GPU detected. Running on CPU.")
except RuntimeError as e:
    print(e)

# ==========================================
# 2. Data Loading & Preparation
# ==========================================
def load_data(data_path):
    print("Loading dataset...")
    X_train = np.load(f'{data_path}/X_train.npy')
    X_test = np.load(f'{data_path}/X_test.npy')
    X_val = np.load(f'{data_path}/X_val.npy')
    y_train = np.load(f'{data_path}/y_train.npy')
    y_test = np.load(f'{data_path}/y_test.npy')
    y_val = np.load(f'{data_path}/y_val.npy')

    X_train, X_test, X_val = X_train[..., np.newaxis], X_test[..., np.newaxis], X_val[..., np.newaxis]
    y_train, y_test, y_val = y_train[..., np.newaxis], y_test[..., np.newaxis], y_val[..., np.newaxis]
    print("Dataset loaded!")
    return X_train, X_test, X_val, y_train, y_test, y_val

def prepare_dataset(data_path):
    X_train, X_test, X_val, y_train, y_test, y_val = load_data(data_path)
    scaler = StandardScaler()
    
    n_train, t_steps, n_feats, _ = X_train.shape
    scaler.fit(X_train.reshape(-1, n_feats))

    for X_split in [X_train, X_test, X_val]:
        n, t, f, _ = X_split.shape
        X_flat_scaled = scaler.transform(X_split.reshape(-1, f))
        X_split[:] = X_flat_scaled.reshape(n, t, f, 1)

    return X_train, y_train, X_val, y_val, X_test, y_test

DATA_PATH = "/home/ec.gpu/Desktop/Soumen/Dataset/kws/data_npy"
X_train, y_train, X_val, y_val, X_test, y_test = prepare_dataset(DATA_PATH)

# ==========================================
# 3. Global Constants & Hyperparameter Space
# ==========================================
CLASSES = 10
EPOCHS = 300
BATCH_SIZE = 64
PATIENCE = 5

# Master Parameter Lists
L_CONV = [1, 2, 3, 4]
L_FILT = [16, 32]
L_KERN = [(3, 3), (5, 5)]
L_FC   = [1, 2, 3]
L_BN   = [True, False]
L_DROP = [True, False]
SPACE_LISTS = [L_CONV, L_FILT, L_KERN, L_FC, L_BN, L_DROP]

HPARAM_SPACE = [
    {'conv_layers': cl, 'filters': f, 'kernel_size': ks, 'fc_layers': fc, 'use_bn': bn, 'use_dropout': do}
    for cl, f, ks, fc, bn, do in itertools.product(*SPACE_LISTS)
]
print(f"Generated discrete hyperparameter space: {len(HPARAM_SPACE)} configurations.")

INITIAL_SAMPLES = 30 
MAX_ITERATIONS = 30
CANDIDATE_BATCH = len(HPARAM_SPACE)

# ==========================================
# 4. DSCNN KWS Model
# ==========================================
def create_kws_model(conv_layers, filters, kernel_size, fc_layers, use_bn, use_dropout, depth_multiplier=1):
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
        model.add(SeparableConv2D(f_out, (3, 3), depth_multiplier=depth_multiplier, padding="same", activation="relu"))
        if use_bn: model.add(BatchNormalization())
            
        if model.output_shape[1] > 2 and model.output_shape[2] > 2:
            model.add(MaxPooling2D(pool_size=(2, 2), padding='same'))

    # Dynamic scaling based on layers
    curr_filters = filters
    for _ in range(conv_layers):
        depthwise_separable_block(curr_filters)
        curr_filters = min(curr_filters * 2, 512)

    model.add(Flatten())

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

    model.add(Dense(CLASSES, activation='softmax'))
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def compute_model_size_mb(model):
    return (model.count_params() * 4) / (1024 ** 2)

def evaluate_individual(hparams):
    tf.keras.backend.clear_session()
    model = create_kws_model(**hparams)
    es = EarlyStopping(monitor='val_loss', patience=PATIENCE, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_val, y_val), 
                        epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=[es], verbose=0)
    
    val_acc = history.history['val_accuracy'][-1]
    size_mb = compute_model_size_mb(model)
    print(f"  -> Eval: Acc={val_acc:.4f}, Size={size_mb:.2f}MB")
    return -val_acc, size_mb

# ==========================================
# 5. Initialization Strategies
# ==========================================
def map_qmc_to_hparams(samples):
    """Maps continuous [0,1] samples to discrete hyperparameter grid."""
    selected = []
    for row in samples:
        hp = {
            'conv_layers': L_CONV[int(row[0] * len(L_CONV))],
            'filters': L_FILT[int(row[1] * len(L_FILT))],
            'kernel_size': L_KERN[int(row[2] * len(L_KERN))],
            'fc_layers': L_FC[int(row[3] * len(L_FC))],
            'use_bn': L_BN[int(row[4] * len(L_BN))],
            'use_dropout': L_DROP[int(row[5] * len(L_DROP))]
        }
        selected.append(hp)
    return selected

def load_oasi_archive(num_samples, file_path="OASI_init.xlsx"):
    """Extracts top solutions from previous Simulated Annealing archive."""
    try:
        df = pd.read_excel(file_path)
        df = df.sort_values(by=['Accuracy', 'Model Size'], ascending=[False, True]) # Sort for best
        
        oasi_samples = []
        for _, row in df.head(num_samples * 2).iterrows(): # Pull extra in case of duplicates
            try:
                # Map extracted values to nearest grid equivalents to prevent out-of-bounds errors
                k_size = ast.literal_eval(row['Kernel Size']) if isinstance(row['Kernel Size'], str) else row['Kernel Size']
                
                hp = {
                    'conv_layers': min(L_CONV, key=lambda x: abs(x - int(row['Conv Layers']))),
                    'filters': min(L_FILT, key=lambda x: abs(x - int(row['Filters']))),
                    'kernel_size': k_size if k_size in L_KERN else (3,3),
                    'fc_layers': random.choice(L_FC), # Mapping discrete dense blocks
                    'use_bn': bool(row['Batch Normalization']),
                    'use_dropout': bool(row['Dropout'])
                }
                if hp not in oasi_samples:
                    oasi_samples.append(hp)
                if len(oasi_samples) == num_samples: break
            except Exception as e:
                continue
        
        # Pad with random if archive is smaller than initial sample size
        while len(oasi_samples) < num_samples:
            oasi_samples.append(random.choice(HPARAM_SPACE))
            
        print(f"Loaded {num_samples} samples from OASI archive.")
        return oasi_samples
        
    except FileNotFoundError:
        print(f"OASI file {file_path} not found. Falling back to Random initialization.")
        return [random.choice(HPARAM_SPACE) for _ in range(num_samples)]

def get_initial_samples(method, num_samples):
    print(f"--- Initializing via {method} ---")
    if method == 'Random':
        return random.sample(HPARAM_SPACE, num_samples)
    elif method == 'LHS':
        sampler = qmc.LatinHypercube(d=6)
        return map_qmc_to_hparams(sampler.random(n=num_samples))
    elif method == 'Sobol':
        sampler = qmc.Sobol(d=6, scramble=True)
        # Power of 2 is optimal for Sobol, but works for arbitrary num_samples
        return map_qmc_to_hparams(sampler.random(n=num_samples))
    elif method == 'OASI':
        return load_oasi_archive(num_samples)

# ==========================================
# 6. MOBO Logic 
# ==========================================
class SurrogateManager:
    def __init__(self):
        self.gps = [GaussianProcessRegressor(kernel=Matern(nu=2.5), normalize_y=True, n_restarts_optimizer=5) for _ in range(2)]
        self.is_fitted = False

    def update(self, X, Y):
        for i, gp in enumerate(self.gps): gp.fit(X, Y[:, i])
        self.is_fitted = True

    def predict(self, X):
        means, stds = [], []
        for gp in self.gps:
            mean, std = gp.predict(X, return_std=True)
            means.append(mean)
            stds.append(std)
        return np.stack(means, axis=1), np.stack(stds, axis=1)

def hparams_to_vector(hp):
    idx = HPARAM_SPACE.index(hp)
    return np.array([idx / (len(HPARAM_SPACE) - 1.0)])

def vector_to_hparams(vec):
    idx = int(round(vec[0] * (len(HPARAM_SPACE) - 1)))
    return HPARAM_SPACE[idx]

def run_mobo(init_method, output_dir):
    hparams_history = get_initial_samples(init_method, INITIAL_SAMPLES)
    X_vec = np.array([hparams_to_vector(hp) for hp in hparams_history])
    Y_objs = np.array([evaluate_individual(hp) for hp in hparams_history])
    
    surrogate_manager = SurrogateManager()
    all_evaluations_log = []
    
    for i in range(INITIAL_SAMPLES):
        rec = {'Init_Method': init_method, 'Iteration': 0, 'Accuracy': -Y_objs[i,0], 'Size_MB': Y_objs[i,1], **hparams_history[i]}
        all_evaluations_log.append(rec)

    plot_dir = os.path.join(output_dir, f"plots_{init_method}")
    os.makedirs(plot_dir, exist_ok=True)

    for it in range(MAX_ITERATIONS):
        print(f"\n[{init_method}] ----- MOBO Iteration {it+1}/{MAX_ITERATIONS} -----")
        surrogate_manager.update(X_vec, Y_objs)
        
        candidates_vec = np.array([hparams_to_vector(hp) for hp in HPARAM_SPACE])
        mu, sigma = surrogate_manager.predict(candidates_vec)
        
        front_indices = NonDominatedSorting().do(Y_objs, only_non_dominated_front=True)
        pareto_front = Y_objs[front_indices]
        
        ref_point = np.max(Y_objs, axis=0) * 1.1 
        hv_indicator = HV(ref_point=ref_point)
        current_hv = hv_indicator.do(pareto_front)

        ehvi_vals = []
        for i in range(CANDIDATE_BATCH):
            samples = np.random.normal(mu[i], sigma[i], size=(100, 2))
            hv_improvements = []
            for sample in samples:
                combined_front = np.vstack([pareto_front, sample])
                front_idx_comb = NonDominatedSorting().do(combined_front, only_non_dominated_front=True)
                new_hv = hv_indicator.do(combined_front[front_idx_comb])
                hv_improvements.append(max(0, new_hv - current_hv))
            ehvi_vals.append(np.mean(hv_improvements))

        best_idx = np.argmax(ehvi_vals)
        x_next_vec = candidates_vec[best_idx]
        hp_next = vector_to_hparams(x_next_vec)
        
        if hp_next in hparams_history:
            print(f"Skipping already evaluated sample: {hp_next}")
            continue 
        
        print(f"Evaluating next sample (max EHVI): {hp_next}")
        obj_next = evaluate_individual(hp_next)
        
        X_vec = np.vstack([X_vec, x_next_vec])
        Y_objs = np.vstack([Y_objs, obj_next])
        hparams_history.append(hp_next)
        
        rec = {'Init_Method': init_method, 'Iteration': it + 1, 'Accuracy': -obj_next[0], 'Size_MB': obj_next[1], **hp_next}
        all_evaluations_log.append(rec)

    final_front_indices = NonDominatedSorting().do(Y_objs, only_non_dominated_front=True)
    pareto_solutions = [(hparams_history[i], Y_objs[i]) for i in final_front_indices]
    
    return pareto_solutions, pd.DataFrame(all_evaluations_log)

# ==========================================
# 7. Main Execution Loop
# ==========================================
if __name__ == "__main__":
    INIT_METHODS = ['Random', 'LHS', 'Sobol', 'OASI']
    BASE_OUT_DIR = "mobo_results"
    os.makedirs(BASE_OUT_DIR, exist_ok=True)
    
    overall_start_time = time.time()
    
    for method in INIT_METHODS:
        print(f"\n{'='*50}\nSTARTING INITIALIZATION METHOD: {method}\n{'='*50}")
        start_time = time.time()
        
        pareto_sols, all_evaluations_df = run_mobo(method, BASE_OUT_DIR)
        
        # Save All Evaluations
        all_eval_path = os.path.join(BASE_OUT_DIR, f"all_evaluation_mobo_{method}.xlsx")
        all_evaluations_df.to_excel(all_eval_path, index=False)
        print(f"✔️ All evaluation data for {method} saved to '{all_eval_path}'")
        
        # Save Pareto Front
        if pareto_sols:
            pareto_records = []
            for hp, obj in pareto_sols:
                rec = {'Accuracy': -obj[0], 'Size_MB': obj[1], **hp}
                pareto_records.append(rec)
            
            df_pareto = pd.DataFrame(pareto_records)
            pareto_path = os.path.join(BASE_OUT_DIR, f"mobo_pareto_{method}.csv")
            df_pareto.to_csv(pareto_path, index=False)
            print(f"✔️ Final Pareto set for {method} saved to '{pareto_path}'")
            
        print(f"Completed {method} in {(time.time() - start_time) / 60:.2f} minutes.")
        
    print(f"\nAll initialization runs finished. Total time: {(time.time() - overall_start_time) / 60:.2f} minutes.")