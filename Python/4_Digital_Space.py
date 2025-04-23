# -*- coding: utf-8 -*-
"""
Script to showcase digital spaces.

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
from sklearn import preprocessing
from sklearn import decomposition
import gpflow
from matplotlib import pyplot as plt
from matplotlib import cm

# =============================================================================
# Configuration
# =============================================================================

# Path to Datasets folder
datasetsPath=r'C:\Users\dinis\Desktop\DeepBayesian\Python\Datasets'
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
plt.rcParams['axes.titlesize']=10
plt.rcParams['axes.labelsize']=10
plt.rcParams['xtick.labelsize']=9
plt.rcParams['ytick.labelsize']=9
plt.rcParams['font.size']=9
plt.rcParams["savefig.pad_inches"]=0.02

# PCI space
SP_Space=np.array([]).reshape(0,61*3)
for aux in ['Density','Viscosity','T_melting']:
    # Define normalization
    featureNorm='Log+bStand'
    labelNorm='LogStand'
    # Open the Train-Val-Test sets
    aux1 = np.load(os.path.join(datasetsPath,'X_Train_'+aux+'.npy'))
    aux2 = np.load(os.path.join(datasetsPath,'X_Val_'+aux+'.npy'))
    aux3 = np.load(os.path.join(datasetsPath,'X_Test_'+aux+'.npy'))
    if aux!='T_melting':
        aux1=aux1[:,:-1]
        aux2=aux2[:,:-1]
        aux3=aux3[:,:-1]
    SP_Space=np.concatenate((SP_Space,aux1,aux2,aux3),axis=0)
SP_Space=np.unique(SP_Space,axis=0)

# Perform PCA
PCA=decomposition.PCA(n_components=2,svd_solver='full').fit(SP_Space)
# PCA=decomposition.KernelPCA(n_components=2,kernel='rbf').fit(SP_Space)
SP_Matrix=np.concatenate((X_Train,X_Val,X_Test),axis=0)
if target!='T_melting':
    SP_Matrix_PCA=PCA.transform(SP_Matrix[:,:-1].copy())
else:
    SP_Matrix_PCA=PCA.transform(SP_Matrix.copy())
Y=np.concatenate((Y_Train_Pred,Y_Val_Pred,Y_Test_Pred),axis=0)

# Plot Surface
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(projection='3d',computed_zorder=False)
ax.set_proj_type('persp',focal_length=0.2)
if target=='Viscosity':
    minZ=Y.reshape(-1,).min()
    ax.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
            minZ*np.ones((SP_Matrix_PCA.shape[0],)),'.',c='gray',zorder=4.3)
    surf=ax.plot_trisurf(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
                         np.log(Y.reshape(-1,)),
                         cmap=cm.jet,antialiased=False,alpha=0.8,
                         zorder=4.4)
else:
    minZ=Y.reshape(-1,).min()
    ax.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
            minZ*np.ones((SP_Matrix_PCA.shape[0],)),'.',c='gray',zorder=4.3)
    surf=ax.plot_trisurf(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],Y.reshape(-1,),
                         cmap=cm.jet,antialiased=False,alpha=0.8,
                         zorder=4.4)
ax.set_xlabel('PCA Var. #1',weight='bold')
ax.set_ylabel('PCA Var. #2',weight='bold')
if target=='Density':
    ax.set_zlabel('Pred. Density',weight='bold')
if target=='Viscosity':
    ax.set_zlabel('Pred. log(Viscosity)',weight='bold')
if target=='T_melting':
    ax.set_zlabel('Pred. Melting Temp.',weight='bold')
ax.set_box_aspect(None,zoom=0.8)
plt.show()

