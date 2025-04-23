# -*- coding: utf-8 -*-
"""
Script to train a GP for a given dataset.

Sections:
    . Imports
    . Configuration
    . Main Functions
        . normalize()
        . buildGP()
        . gpPredict()
    . Main Script
    . Plots

Authors: João Santos & Dinis Abranches
Last Edit: 2025-04-01
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import time
import warnings

# Specific
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import gpflow
from matplotlib import pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Path to Datasets folder
datasetsPath=r'/path/to/Python/Datasets'
# Target variable
target='T_melting' # 'Density', 'Viscosity', 'T_melting'
# Options: None, Standardization, MinMax, LogStand, Log+bStand
featureNorm = 'Log+bStand'
# Options: None, Standardization, MinMax, LogStand, Log+bStand
labelNorm = 'LogStand'
# gpConfig
gpConfig = {'kernel': 'Matern32',
            'useWhiteKernel': True}

# =============================================================================
# Main Functions
# =============================================================================

def normalize(X, skScaler=None, method='Standardization', reverse=False):
    """
    Normalize or unnormalize input array using the specified method and scaler.

    Parameters
    ----------
    X : numpy array
        Array to be normalized. If dim > 1, array is normalized column-wise.
    skScaler : scikit-learn preprocessing object or None
        Preprocessing object previously fitted to data. If None, it is fitted
        to X.
    method : string, optional
        Normalization method: 'Standardization', 'MinMax', 'LogStand',
        'Log+bStand'.
        Default: 'Standardization'
    reverse : bool, optional
        Whether to normalize (False) or unnormalize (True) the input array.
        Default: False

    Returns
    -------
    X : numpy array
        Normalized or unnormalized version of X.
    skScaler : scikit-learn preprocessing object
        Fitted preprocessing object.
    """
    if X.ndim == 1:
        X = X.reshape((-1, 1))
        warnings.warn('Input was reshaped to (N, 1).')
    if skScaler is None:
        if method == 'Standardization':
            skScaler = preprocessing.StandardScaler().fit(X)
        elif method == 'MinMax':
            skScaler = preprocessing.MinMaxScaler().fit(X)
        elif method == 'LogStand':
            skScaler = preprocessing.StandardScaler().fit(np.log(X))
        elif method == 'Log+bStand':
            skScaler = preprocessing.StandardScaler().fit(np.log(X + 1e-3))
        else:
            raise ValueError('Unrecognized method.')
    if reverse:
        X = skScaler.inverse_transform(X)
        if method == 'LogStand':
            X = np.exp(X)
        elif method == 'Log+bStand':
            X = np.exp(X) - 1e-3
    else:
        if method == 'LogStand':
            X = skScaler.transform(np.log(X))
        elif method == 'Log+bStand':
            X = skScaler.transform(np.log(X + 1e-3))
        else:
            X = skScaler.transform(X)
    return X, skScaler

def buildGP(X_Train, Y_Train, gpConfig={}):
    """
    Build and fit a GP model using the training data provided.

    Parameters
    ----------
    X_Train : numpy array (N, K)
        Training features.
    Y_Train : numpy array (N, 1)
        Training labels.
    gpConfig : dictionary, optional
        Configuration of the GP. Default is {}.

    Returns
    -------
    model : gpflow.models.GPR object
        GP model.
    """
    kernel = gpConfig.get('kernel', 'RQ')
    useWhiteKernel = gpConfig.get('useWhiteKernel', True)
    trainLikelihood = gpConfig.get('trainLikelihood', False)    
    if kernel=='RBF':
         gpKernel=gpflow.kernels.SquaredExponential()
    elif kernel=='RQ':
         gpKernel=gpflow.kernels.RationalQuadratic()
    elif kernel=='Matern32':
         gpKernel=gpflow.kernels.Matern32()
    elif kernel=='Matern52':
         gpKernel=gpflow.kernels.Matern52()
    else:
        raise ValueError('Unrecognized kernel.')
    if useWhiteKernel:
        gpKernel += gpflow.kernels.White()

    model = gpflow.models.GPR(data=(X_Train, Y_Train), kernel=gpKernel,
                              noise_variance=1e-5)
    gpflow.utilities.set_trainable(model.likelihood.variance, trainLikelihood)
    optimizer = gpflow.optimizers.Scipy()
    result = optimizer.minimize(model.training_loss, model.trainable_variables,
                                method='L-BFGS-B')
    if not result.success:
        warnings.warn('GP optimizer failed to converge.')
    return model

def gpPredict(model, X):
    """
    Return the prediction and standard deviation of the GP model on the X data
    provided.

    Parameters
    ----------
    model : gpflow.models.GPR object
        GP model.
    X : numpy array (N, K)
        Features for prediction.

    Returns
    -------
    Y : numpy array (N, 1)
        GP predictions.
    STD : numpy array (N, 1)
        GP standard deviations.
    """
    GP_Mean, GP_Var = model.predict_f(X)
    Y = GP_Mean.numpy()
    STD = np.sqrt(GP_Var.numpy())
    return Y, STD

# =============================================================================
# Main Script
# =============================================================================

# Initialize timer
ti=time.time()

# Open the Train-Val-Test sets
X_Train = np.load(os.path.join(datasetsPath,'X_Train_'+target+'.npy'))
X_Val = np.load(os.path.join(datasetsPath,'X_Val_'+target+'.npy'))
X_Test = np.load(os.path.join(datasetsPath,'X_Test_'+target+'.npy'))
Y_Train = np.load(os.path.join(datasetsPath,'Y_Train_'+target+'.npy'))
Y_Val = np.load(os.path.join(datasetsPath,'Y_Val_'+target+'.npy'))
Y_Test = np.load(os.path.join(datasetsPath,'Y_Test_'+target+'.npy'))

# Normalize the features
if featureNorm is not None:
    X_Train_N, skScaler_X_Train = normalize(X_Train, method=featureNorm)
    X_Val_N, _ = normalize(X_Val, skScaler=skScaler_X_Train,
                           method=featureNorm)
    X_Test_N, _ = normalize(X_Test, skScaler=skScaler_X_Train,
                            method=featureNorm)
else:
    X_Train_N, X_Val_N, X_Test_N = X_Train, X_Val, X_Test
# Normalize the labels
if labelNorm is not None:
    Y_Train_N, skScaler_Y_Train = normalize(Y_Train, method=labelNorm)
else:
    Y_Train_N = Y_Train

# Build and train the GP model
model = buildGP(X_Train_N, Y_Train_N, gpConfig=gpConfig)

# Get predictions and standard deviations for train, val, and test data
Y_Train_Pred_N, STD_Train = gpPredict(model, X_Train_N)
Y_Val_Pred_N, STD_Val = gpPredict(model, X_Val_N)
Y_Test_Pred_N, STD_Test = gpPredict(model, X_Test_N)

# Unnormalize predictions
if labelNorm is not None:
    Y_Train_Pred, _ = normalize(Y_Train_Pred_N, skScaler=skScaler_Y_Train,
                                method=labelNorm, reverse=True)
    Y_Val_Pred, _ = normalize(Y_Val_Pred_N, skScaler=skScaler_Y_Train,
                              method=labelNorm, reverse=True)
    Y_Test_Pred, _ = normalize(Y_Test_Pred_N, skScaler=skScaler_Y_Train,
                              method=labelNorm, reverse=True)
else:
    Y_Train_Pred, Y_Val_Pred, Y_Test_Pred = \
        Y_Train_Pred_N, Y_Val_Pred_N, Y_Test_Pred_N

# =============================================================================
# Plots
# =============================================================================

# Pyplot Configuration
plt.rcParams['figure.dpi']=600
plt.rcParams['font.family']='serif'
plt.rcParams['font.serif']='Arial'
plt.rcParams['font.weight']='bold'
plt.rcParams['axes.titlesize']=9
plt.rcParams['axes.labelsize']=9
plt.rcParams['xtick.labelsize']=8
plt.rcParams['ytick.labelsize']=8
plt.rcParams['font.size']=8
plt.rcParams["savefig.pad_inches"]=0.02

# Predictions Scatter Plot
MAE_Train = metrics.mean_absolute_error(Y_Train, Y_Train_Pred)
MAE_Val = metrics.mean_absolute_error(Y_Val, Y_Val_Pred)
MAE_Test = metrics.mean_absolute_error(Y_Test, Y_Test_Pred)
if target == 'Viscosity':
    MAE_Train = metrics.mean_absolute_error(np.log(Y_Train),
                                            np.log(Y_Train_Pred))
    MAE_Val = metrics.mean_absolute_error(np.log(Y_Val), np.log(Y_Val_Pred))
    MAE_Test = metrics.mean_absolute_error(np.log(Y_Test), np.log(Y_Test_Pred))
    R2_Train = metrics.r2_score(np.log(Y_Train), np.log(Y_Train_Pred))
    R2_Val = metrics.r2_score(np.log(Y_Val), np.log(Y_Val_Pred))
    R2_Test = metrics.r2_score(np.log(Y_Test), np.log(Y_Test_Pred))
    plt.figure(figsize=(2.3, 2))
    plt.loglog(Y_Train, Y_Train_Pred, 'ob', markersize=3)
    plt.loglog(Y_Val, Y_Val_Pred, 'sr', markersize=2)
    plt.loglog(Y_Test, Y_Test_Pred, '^b', markersize=2)
    
else:
    R2_Train = metrics.r2_score(Y_Train, Y_Train_Pred)
    R2_Val = metrics.r2_score(Y_Val, Y_Val_Pred)
    R2_Test = metrics.r2_score(Y_Test, Y_Test_Pred)
    plt.figure(figsize=(2.3, 2))
    plt.plot(Y_Train, Y_Train_Pred, 'ok', markersize=3)
    plt.plot(Y_Val, Y_Val_Pred, 'sr', markersize=2)
    plt.plot(Y_Test, Y_Test_Pred, '^b', markersize=2)

lims = [
    np.min([plt.gca().get_xlim()[0], plt.gca().get_ylim()][0]),
    np.max([plt.gca().get_xlim()[1], plt.gca().get_ylim()[1]])
]

# Plot the diagonal line
plt.plot(lims, lims, color='k', linestyle='--', linewidth=1)

# Set labels and other plot configurations
if target=='Density':
    units='/g$\cdot$mL$^{-1}$'
    plt.xlabel('Exp. Density /g$\cdot$mL$^{-1}$',weight='bold')
    plt.ylabel('Pred. Density /g$\cdot$mL$^{-1}$',weight='bold')
elif target=='Viscosity':
    units='/cP'
    plt.xlabel('Exp. Viscosity /cP',weight='bold')
    plt.ylabel('Pred. Viscosity /cP',weight='bold')
elif target=='T_melting':
    units='/K'
    plt.xlabel('Exp. Melting Temp. /K',weight='bold')
    plt.ylabel('Pred. Melting Temp. /K',weight='bold')

# Add text annotations for MAE and R^2
if target=='Viscosity':
    plt.text(0.03, 0.92, f'MALE = {MAE_Train:.2f}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='k')
    plt.text(0.03, 0.84, f'MALE = {MAE_Val:.2f}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='r')
    plt.text(0.03, 0.76, f'MALE = {MAE_Test:.2f}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='b')
    plt.text(0.97, 0.18, f'$R^2$ = {R2_Train:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='k')
    plt.text(0.97, 0.10, f'$R^2$ = {R2_Val:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='r')
    plt.text(0.97, 0.02, f'$R^2$ = {R2_Test:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='b')
elif target=='T_melting':
    plt.text(0.03, 0.92, f'MAE = {MAE_Train:.0f} {units}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='k')
    plt.text(0.03, 0.84, f'MAE = {MAE_Val:.0f} {units}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='r')
    plt.text(0.03, 0.76, f'MAE = {MAE_Test:.0f} {units}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='b')
    plt.text(0.97, 0.18, f'$R^2$ = {R2_Train:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='k')
    plt.text(0.97, 0.10, f'$R^2$ = {R2_Val:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='r')
    plt.text(0.97, 0.02, f'$R^2$ = {R2_Test:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='b')
else:
    plt.text(0.03, 0.92, f'MAE = {MAE_Train:.2f} {units}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='k')
    plt.text(0.03, 0.84, f'MAE = {MAE_Val:.2f} {units}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='r')
    plt.text(0.03, 0.76, f'MAE = {MAE_Test:.2f} {units}',
             horizontalalignment='left', transform=plt.gca().transAxes,
             color='b')
    plt.text(0.97, 0.18, f'$R^2$ = {R2_Train:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='k')
    plt.text(0.97, 0.10, f'$R^2$ = {R2_Val:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='r')
    plt.text(0.97, 0.02, f'$R^2$ = {R2_Test:.2f}',
             horizontalalignment='right', transform=plt.gca().transAxes,
             color='b')
# Show the plot
plt.show()

