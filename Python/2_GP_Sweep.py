# -*- coding: utf-8 -*-
"""
Script to sweep GP configurations.

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
import warnings

# Specific
import numpy as np
from sklearn import metrics
from sklearn import preprocessing
import gpflow
from tqdm import tqdm

# =============================================================================
# Configuration
# =============================================================================

# Path to Datasets folder
datasetsPath=r'/path/to/Python/Datasets'
# Target variable
target='T_melting' # 'Density', 'Viscosity', 'T_melting'
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

# Open the Train-Val-Test sets
X_Train = np.load(os.path.join(datasetsPath,'X_Train_'+target+'.npy'))
X_Val = np.load(os.path.join(datasetsPath,'X_Val_'+target+'.npy'))
Y_Train = np.load(os.path.join(datasetsPath,'Y_Train_'+target+'.npy'))
Y_Val = np.load(os.path.join(datasetsPath,'Y_Val_'+target+'.npy'))

# Initialize results container
results = np.zeros((3, 3))
# Initialize loops
for n, featureNorm in enumerate(tqdm([None, 'Standardization', 'Log+bStand'])):
    for k, labelNorm in enumerate(tqdm([None, 'Standardization', 'LogStand'])):
        # Normalize the features
        if featureNorm is not None:
            X_Train_N, skScaler_X_Train = normalize(X_Train,
                                                    method=featureNorm)
            X_Val_N, __ = normalize(X_Val, skScaler=skScaler_X_Train,
                                    method=featureNorm)
        else:
            X_Train_N, X_Val_N= X_Train, X_Val
        # Normalize the labels
        if labelNorm is not None:
            Y_Train_N, skScaler_Y_Train = normalize(Y_Train, method=labelNorm)
        else:
            Y_Train_N = Y_Train
        # Build and train the GP model
        model = buildGP(X_Train_N, Y_Train_N, gpConfig=gpConfig)
        # Get predictions for validation set
        Y_Val_Pred_N, STD_Val = gpPredict(model, X_Val_N)
        # Unnormalize predictions
        if labelNorm is not None:
            Y_Val_Pred, __ = normalize(Y_Val_Pred_N, skScaler=skScaler_Y_Train,
                                       method=labelNorm, reverse=True)
        else:
            Y_Val_Pred = Y_Val_Pred_N
        # Compute Val R^2
        if target == 'Viscosity':
            try:
                R2_val = metrics.r2_score(np.log(Y_Val),
                                          np.log(Y_Val_Pred))
            except:
                R2_val = np.array([-100])
        else:
            R2_val = metrics.r2_score(Y_Val, Y_Val_Pred)
        # Update results container
        results[n,k]=R2_val.copy()
        


