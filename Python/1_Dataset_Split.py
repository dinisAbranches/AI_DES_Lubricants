# -*- coding: utf-8 -*-
"""
Script to split the database into training, validation, and testing sets using
stratified sampling.

Sections:
    . Imports
    . Configuration
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

# Specific
import numpy
import pandas
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt

# =============================================================================
# Configuration
# =============================================================================

# Path to Datasets folder
datasetsPath=r'/path/to/Python/Datasets'

# =============================================================================
# Main Script
# =============================================================================

# Generate plot containers
plot_List=[]
# Loop over datasets
for var in ['Density','Viscosity','T_melting']:
    # Define var name
    if var=='Density': name='Density /g/cm^3'
    elif var =='Viscosity': name='Viscosity /cP'
    elif var=='T_melting': name='T_melting /K'
    # Read SP dataset
    data_sigma=pandas.read_csv(os.path.join(datasetsPath,'Dataset_SP.csv'))
    # Read var dataset
    data_var=pandas.read_csv(os.path.join(datasetsPath,'Dataset_'+var+'.csv'))
    # Concatenate datasets
    data_info=pandas.concat([data_sigma,data_var],axis=1)
    info_data=data_var[name].tolist()
    # Extract relevant columns
    component_1_values=data_var['Component #1'].tolist()
    component_2_values=data_var['Component #2'].tolist()
    component_3_values=data_var['Component #3'].tolist()
    # Print unique combinations
    print(var+' - Number of unique DES combinations: '\
          +str(data_var.iloc[:,:3].drop_duplicates().shape[0]))
    x_1_values=data_var['x1'].tolist()
    x_2_values=data_var['x2'].tolist()
    x_3_values=data_var['x3'].tolist()
    if var!='T_melting': temperature=data_var['T /K'].tolist()
    # Get list of available sigma profile names
    availableList=data_sigma['TZVP'].tolist()
    # Associate compound names in data_var to sigma profiles in data_sigma
    def append_values(component_values,x_values):
        component_sigmas=numpy.zeros((len(component_values),61))
        for n,component in enumerate(component_values):
            if component in availableList:
                value=float(x_values[n])
                SP=data_sigma[data_sigma['TZVP'] == component]
                SP=SP.iloc[0,1:].to_numpy('double')
            else:
                SP=numpy.zeros((1,61))
                value=0
            component_sigmas[n,:]=SP*value
        return component_sigmas
    component_sigmas_1=append_values(component_1_values,x_1_values)
    component_sigmas_2=append_values(component_2_values,x_2_values)
    component_sigmas_3=append_values(component_3_values,x_3_values)
    # Define main X variable
    X=numpy.concatenate((component_sigmas_1,component_sigmas_2, 
                      component_sigmas_3),axis=1)
    if var!='T_melting':
        X=numpy.concatenate((X,numpy.array(temperature).reshape(-1, 1)),axis=1)
    Y=numpy.array(info_data).reshape(-1, 1)
    if var=='Viscosity':
        Y_aux=numpy.log(Y)
    else:
        Y_aux=Y
    # Train/Test Stratification
    for n in range(1,100):
        # Bin Y using n bins
        stratifyVector=pandas.cut(Y_aux.reshape(-1,),n,labels=False)
        # Define isValid (all bins have at least 5 values)
        isValid=True
        # Check that all bins have at least 5 values
        for k in range(n):
            if numpy.count_nonzero(stratifyVector==k)<5:
                isValid=False
        #If isValid is false, n is too large; nBins must be the previous iteration
        if not isValid:
            nBins=n-1
            break
    # Generate vector for stratified splitting based on labels
    stratifyVector=pandas.cut(Y_aux.reshape(-1,),nBins,labels=False)
    # Perform Train/Test splitting
    X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y, 
                                                     train_size=0.9,
                                                     random_state=42,
                                                     stratify=stratifyVector)
    # Validation Stratification
    if var=='Viscosity':
        Y_Train_aux=numpy.log(Y_Train)
    else:
        Y_Train_aux=Y_Train
    for n in range(1,100):
        # Bin Y using n bins
        stratifyVector=pandas.cut(Y_Train_aux.reshape(-1,),n,labels=False)
        # Define isValid (all bins have at least 5 values)
        isValid=True
        # Check that all bins have at least 5 values
        for k in range(n):
            if numpy.count_nonzero(stratifyVector==k)<5:
                isValid=False
        #If isValid is false, n is too large; nBins must be the previous iteration
        if not isValid:
            nBins=n-1
            break
    # Generate vector for stratified splitting based on labels
    stratifyVector=pandas.cut(Y_Train_aux.reshape(-1,),nBins,labels=False)
    # Perform Train/Test splitting
    X_Train,X_Val,Y_Train,Y_Val = train_test_split(X_Train, Y_Train, 
                                                   train_size=0.78,
                                                   random_state=42,
                                                   stratify=stratifyVector)
    #Save Tain, Test and Validation sets
    numpy.save(os.path.join(datasetsPath,'X_Train_'+var+'.npy'),X_Train)
    numpy.save(os.path.join(datasetsPath,'X_Val_'+var+'.npy'),X_Val)
    numpy.save(os.path.join(datasetsPath,'X_Test_'+var+'.npy'),X_Test)
    numpy.save(os.path.join(datasetsPath,'Y_Train_'+var+'.npy'),Y_Train)
    numpy.save(os.path.join(datasetsPath,'Y_Val_'+var+'.npy'),Y_Val)
    numpy.save(os.path.join(datasetsPath,'Y_Test_'+var+'.npy'),Y_Test)
    # Print shapes and ranges
    print("Shape of Y_Train:",Y_Train.shape)
    print("Shape of Y_Val:",Y_Val.shape)
    print("Shape of Y_Test:",Y_Test.shape)
    print("Range of Y_Train:",Y_Train.min(),'|',Y_Train.max())
    print("Range of Y_Val:",Y_Val.min(),'|',Y_Val.max())
    print("Range of Y_Test:",Y_Test.min(),'|',Y_Test.max())
    # Add to plot_List
    plot_List.append((Y_Train,Y_Val,Y_Test))
    
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

# Initialize figure
fig,axs=plt.subplots(nrows=1,ncols=3,sharey=True,figsize=(6.5,2.5))
# Loop over compounds
for k,var in enumerate(['Density', 'Viscosity', 'T_melting']):
    if k==1:
        axs[k].hist(plot_List[k][0],
            numpy.exp(numpy.histogram_bin_edges(numpy.log(plot_List[k][0]))),
                    log=True,color='white',edgecolor='black',hatch='xx')
        axs[k].hist(plot_List[k][1],
            numpy.exp(numpy.histogram_bin_edges(numpy.log(plot_List[k][1]))),
                    log=True,color='white',edgecolor='red',hatch='**')
        axs[k].hist(plot_List[k][2],
            numpy.exp(numpy.histogram_bin_edges(numpy.log(plot_List[k][2]))),
                    log=True,color='white',edgecolor='blue',hatch='..')
        axs[k].set_xscale('log')
    else:
        axs[k].hist(plot_List[k][0],log=True,
                    color='white',edgecolor='black',hatch='xx')
        axs[k].hist(plot_List[k][1],log=True,
                    color='white',edgecolor='red',hatch='**')
        axs[k].hist(plot_List[k][2],log=True,
                    color='white',edgecolor='blue',hatch='..')
    if var=='Density': axs[k].set_xlabel('Density /g$\cdot$mL$^{-1}$',
                                         weight='bold')
    elif var =='Viscosity': axs[k].set_xlabel('Viscosity /cP',
                                              weight='bold')
    elif var=='T_melting': axs[k].set_xlabel('Melting Temp. /K',
                                             weight='bold')
    if k==0:
        axs[k].set_ylabel('Count',weight='bold')
plt.tight_layout()
plt.show()
