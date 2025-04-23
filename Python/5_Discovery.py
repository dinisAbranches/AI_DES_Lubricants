# -*- coding: utf-8 -*-
"""
DES discovery.

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
Last Edit: 2024-04-01
"""

# =============================================================================
# Imports
# =============================================================================

# General
import os
import warnings
import copy

# Specific
import numpy as np
import pandas as pd
from tqdm import tqdm
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

# Initialize model container
modelList=[]
skScaler_X_Train_List=[]
skScaler_Y_Train_List=[]
for target in tqdm(['Density','Viscosity','T_melting']):
    # Define normalization
    featureNorm='Log+bStand'
    labelNorm='LogStand'
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
        skScaler_X_Train_List.append(copy.deepcopy(skScaler_X_Train))
    else:
        X_Train_N = X_Train
        skScaler_X_Train_List.append([])
    # Normalize the labels
    if labelNorm is not None:
        Y_Train_N, skScaler_Y_Train = normalize(Y_Train, method=labelNorm)
        skScaler_Y_Train_List.append(copy.deepcopy(skScaler_Y_Train))
    else:
        Y_Train_N = Y_Train
        skScaler_Y_Train_List.append([])
    # Build and train the GP model
    model = buildGP(X_Train_N, Y_Train_N, gpConfig=gpConfig)
    # Append model
    modelList.append(copy.deepcopy(model))

# Load full sigma profile dataset
spDB = pd.read_csv(os.path.join(datasetsPath,'Dataset_SP.csv'))
# Initialize X_Test
X_Test=np.zeros((spDB.shape[0]**2,61*3))
indexList=[]
compositionList=[]
count=0
# Generate sigma profile grid at room temperature, 1:1
for n in range(spDB.shape[0]):
    for k in range(spDB.shape[0]):
        # Retrieve sigma profiles
        SP1=spDB.iloc[n,1:].to_numpy('double')
        SP2=spDB.iloc[k,1:].to_numpy('double')
        # Update X_Test
        X_Test[count,:]=np.concatenate(((SP1*0.5).reshape(1,-1),
                                        (SP2*0.5).reshape(1,-1),
                                        np.zeros((1,61))),axis=1)
        indexList.append([n,k])
        compositionList.append([0.5,0.5])
        count+=1

# Initialize Y_Test_List
Y_Test_List=[]
# Predict 
for n,target in enumerate(['Density','Viscosity','T_melting']):
    # Define normalization
    featureNorm='Log+bStand'
    labelNorm='LogStand'
    # Define X_Test_Aux
    if target!='T_melting':
        X_Test_Aux=np.concatenate((X_Test,
                                  298*np.ones((X_Test.shape[0],1))),axis=1)
    else:
        X_Test_Aux=X_Test
    # Normalize the features
    if featureNorm is not None:
        X_Test_N, __ = normalize(X_Test_Aux, skScaler=skScaler_X_Train_List[n],
                                 method=featureNorm)
    else:
        X_Test_N = X_Test_Aux
    # Get predictions
    Y_Test_N=np.zeros((X_Test_N.shape[0],1))
    for i in tqdm(np.arange(0,Y_Test_N.shape[0]-1000,1000)):
        Y_Test_N[i:i+1000], __ = gpPredict(modelList[n],
                                           X_Test_N[i:i+1000,:])
    # Unnormalize predictions
    if labelNorm is not None:
        Y_Test, __ = normalize(Y_Test_N, skScaler=skScaler_Y_Train_List[n],
                                    method=labelNorm, reverse=True)
    else:
        Y_Test = Y_Test_N
    # Append
    Y_Test_List.append(Y_Test.copy())

# Convert to arrays
Densities=Y_Test_List[0]
Viscosities=Y_Test_List[1]
Temperatures=Y_Test_List[2]
results=np.concatenate((np.array(indexList).reshape(-1,2),
                        np.array(compositionList).reshape(-1,2),
                        Densities,
                        Viscosities,
                        Temperatures),axis=1)

# =============================================================================
# Plots
# =============================================================================

Densities=results[:,4]
Viscosities=results[:,5]
Temperatures=results[:,6]

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

# PCI space
SP_Space=np.array([]).reshape(0,61*3)
for target in tqdm(['Density','Viscosity','T_melting']):
    # Define normalization
    featureNorm='Log+bStand'
    labelNorm='LogStand'
    # Open the Train-Val-Test sets
    aux1 = np.load(os.path.join(datasetsPath,'X_Train_'+target+'.npy'))
    aux2 = np.load(os.path.join(datasetsPath,'X_Val_'+target+'.npy'))
    aux3 = np.load(os.path.join(datasetsPath,'X_Test_'+target+'.npy'))
    if target!='T_melting':
        aux1=aux1[:,:-1]
        aux2=aux2[:,:-1]
        aux3=aux3[:,:-1]
    SP_Space=np.concatenate((SP_Space,aux1,aux2,aux3),axis=0)
SP_Space=np.unique(SP_Space,axis=0)

# Perform PCA
PCA=decomposition.PCA(n_components=2,svd_solver='full').fit(SP_Space)
SP_Space_PCA=PCA.transform(SP_Space.copy())
# PCA=decomposition.KernelPCA(n_components=2,kernel='rbf').fit(SP_Space)
SP_Matrix_PCA=PCA.transform(X_Test.copy())

highlight=np.concatenate((
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[19,264],axis=1))[0]],
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[160,246],axis=1))[0]],
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[264,172],axis=1))[0]],
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[114,246],axis=1))[0]],
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[221,57],axis=1))[0]],
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[99,183],axis=1))[0]],
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[338,137],axis=1))[0]],
    SP_Matrix_PCA[np.where(np.all(results[:,:2]==[326,183],axis=1))[0]]
    ),axis=0)

plt.figure(figsize=(4,3.5))
plt.plot(SP_Space_PCA[:,0],
         SP_Space_PCA[:,1],'.',color='red',alpha=0.2)
plt.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],'.',color='grey',alpha=0.2)
plt.plot(highlight[:,0],highlight[:,1],'*k')
plt.show()

plt.xlabel('PCA Var. #1',weight='bold')
plt.ylabel('PCA Var. #2',weight='bold')
plt.show()

# Plot Surfaces
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(projection='3d',computed_zorder=False)
ax.set_proj_type('persp',focal_length=0.2)
minZ=Densities.reshape(-1,).min()
ax.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
        minZ*np.ones((SP_Matrix_PCA.shape[0],)),'.',c='gray',zorder=4.3)
surf=ax.plot_trisurf(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
                     Densities.reshape(-1,),
                     cmap=cm.jet,antialiased=False,alpha=0.8,zorder=4.4)
ax.set_xlabel('PCA Var. #1',weight='bold')
ax.set_ylabel('PCA Var. #2',weight='bold')
ax.set_zlabel('Pred. Density',weight='bold')
ax.set_box_aspect(None,zoom=0.8)
plt.show()

# Plot Surfaces
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(projection='3d',computed_zorder=False)
ax.set_proj_type('persp',focal_length=0.2)
minZ=Viscosities.reshape(-1,).min()
ax.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
        minZ*np.ones((SP_Matrix_PCA.shape[0],)),'.',c='gray',zorder=4.3)
surf=ax.plot_trisurf(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
                     np.log(Viscosities.reshape(-1,)),
                     cmap=cm.jet,antialiased=False,alpha=0.8,
                     zorder=4.4)
ax.set_xlabel('PCA Var. #1',weight='bold')
ax.set_ylabel('PCA Var. #2',weight='bold')
ax.set_zlabel('Pred. Viscosity',weight='bold')
ax.set_box_aspect(None,zoom=0.8)
plt.show()

# Plot Surfaces
fig=plt.figure(figsize=(10,5))
ax=fig.add_subplot(projection='3d',computed_zorder=False)
ax.set_proj_type('persp',focal_length=0.2)
minZ=Temperatures.reshape(-1,).min()
ax.plot(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
        minZ*np.ones((SP_Matrix_PCA.shape[0],)),'.',c='gray',zorder=4.3)
surf=ax.plot_trisurf(SP_Matrix_PCA[:,0],SP_Matrix_PCA[:,1],
                     Temperatures.reshape(-1,),
                     cmap=cm.jet,antialiased=False,alpha=0.8,zorder=4.4)
ax.set_xlabel('PCA Var. #1',weight='bold')
ax.set_ylabel('PCA Var. #2',weight='bold')
ax.set_zlabel('Pred. Melting Temp.',weight='bold')
ax.set_box_aspect(None,zoom=0.8)
plt.show()



