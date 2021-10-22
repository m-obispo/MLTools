'''
This Python3 script generates molecular descriptors
for MLatom.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from config import *

def genH2O2_Ng(i_file_name, Ng, dR, dTeta=0., dAlpha=0.):
    '''
    Generates MLatom-compatible raw molecular geometries for the H2O2-Ng system.
    --------------------------------------------------------------------------------
    Params:
     i_file_name (str): Name of .xyz or .dat file to be generated.
              Ng (str): Chemical symbol of the noble gas (He, Ne, Ar, Kr, Xe).
            dR (float): Step in R variable (distance between Ng and O-O bond, in
                        angstroms).
         dTeta (float): Step in Theta variable (angle between the two O-O-H planes,
                        in degrees).
    '''
    global D, d, chi, teta1, teta2
    
    with open(i_file_name,'w') as i_file:
        atom = ['O', 'O', 'H', 'H', Ng]
        M = len(atom)
        dsen = d*np.sin(np.radians(chi))
        dcos = d*np.cos(np.radians(chi))
        D2 = D/2

        if dAlpha == 0.:
            alpha_range = [90.]
        else:
            alpha_range = np.arange(0., 180. + dAlpha, dAlpha)

        if dTeta == 0.:
            teta_range = [90.]
        else:
            teta_range = np.arange(0., 360., dTeta)

        for Alpha in alpha_range:
            cos_a = np.cos(np.radians(Alpha))
            sen_a = np.sin(np.radians(Alpha))
            for Teta in teta_range:
                teta2 = teta1 + Teta
                cos_t1 = np.cos(np.radians(teta1))
                cos_t2 = np.cos(np.radians(teta2))
                sen_t1 = np.sin(np.radians(teta1))
                sen_t2 = np.sin(np.radians(teta2))
                for R in np.arange(3., 5.+dR, dR):
                    #             O   O   H            H            Ng
                    x = np.array([0., 0., dsen*sen_t1, dsen*sen_t2, 0.     ])
                    y = np.array([0., 0., dsen*cos_t1, dsen*cos_t2, R*sen_a])
                    z = np.array([D2, -D2, D2 - dcos, - D/2 + dcos, R*cos_a])
                    i_file.write("{}\n\n".format(M))
                    for i in range(M):
                        i_file.write("{}    {:.5f}  {:.5f}  {:.5f}\n".format(atom[i], x[i], y[i], z[i]))


def fetchEq(i_file_name, eq_file_name):
    '''
    Generates MLatom-compatible near-equilibrium configurations from Gaussian logs
    for arbitrary molecular systems.
    --------------------------------------------------------------------------------
    Params:
      i_file_name (str): Name of .log file to be parsed.
     eq_file_name (str): Name of .xyz or .dat file to be generated with equilibrium
                         geometry.
    '''
    global keywords_opt
    atom, x_eq, y_eq, z_eq = [], [], [], []
    c = 0
    with open(i_file_name, 'r') as i_file:
        for line in i_file:
            if c == 730 and keywords_opt[2] not in line:
                linha = line.split()
                atom.append(linha[1])
                x_eq.append(float(linha[3]))
                y_eq.append(float(linha[4]))
                z_eq.append(float(linha[5]))
            else:
                for caju in [kws in line for kws in keywords_opt]:
                    if caju:
                        c += 1

    M = len(atom)

    with open(eq_file_name, 'w') as eq_file:
        eq_file.write("{}\n\n".format(M))
        for i in range(M):
            eq_file.write("{}   {:.3f}  {:.3f}  {:.3f}\n".format(atom[i], x_eq[i], y_eq[i], z_eq[i]))
    
    return {'atom':atom,
            'x':x_eq,
            'y':y_eq,
            'z':z_eq}

def fetchEnergies(i_file_name, E_file_name):
    '''
    Generates MLatom-compatible reference energy values from Gaussian logs for
    arbitrary molecular systems. Currently, only MPn/aug-cc-pVTZ level calculations
    are supported.
    --------------------------------------------------------------------------------
    Params:
      i_file_name (str): Name of .log file to be parsed WITHOUT EXTENSION.
      E_file_name (str): Name of .dat file to be generated with reference energies.
    '''
    global keywords_phf
    e_file = open(E_file_name,'w')
    MP4 = np.zeros(21)
    c = 0
    
    #Energy at R -> inf (theta = 0°)
    with open(i_file_name+"_inf.log",'r') as g:
        for line in g:
            linha = line.split()
            if linha[:3] == keywords_phf:
                # print(linha)
                Einf = float(linha[4])
                
    for k in range(36): #[0,17]: 
        mp4 = []
        with open(i_file_name+"_{}.log".format(k),'r') as i_file:
            for line in i_file:
                linha = line.split()
                if linha[:3] == keywords_phf:
                    mp4.append(float(linha[-1]))
                    e_file.write('{:.5f}\n'.format((float(linha[-1]) - Einf)*219474.6305))
                    c += 1
                    # print('Energy no. {}: {:.5f} Ha'.format(c,float(linha[-1])))
                    
        MP4 = np.vstack((MP4,np.array(mp4)))
    MP4 = MP4[1:,:]
    E_grid = (MP4[1:,:] - Einf)*219474.6305
    e_file.close()
    return MP4

def genFig(Title = '', subTitle = '', xLabel, yLabel):
    fig = plt.figure(figsize = (10, 10), dpi=200)
    plt.suptitle(Title)
    plt.title(subTitle)
    plt.grid()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks()
    plt.yticks()

def correlFig(out_file_name, E_ref_file, E_ML_file, fig_name = None, show = True):
    global keywords_estAcc, statInfo
    o_file = open(out_file_name, 'r')
    c = 0
    statInfo = {}
    for line in o_file:
        linha = line.split()
        # print(c,'\n',linha,'\n')
        if linha[-2:] == keywords_estAcc: c += 1
        if c >= 1 and linha != []:
            if linha[0] == 'MAE': statInfo['MAE'] = float(linha[2])
            if linha[0] == 'MSE': statInfo['MSE'] = float(linha[2])
            if linha[0] == 'RMSE': statInfo['RMSE'] = float(linha[2])
            if linha[0] == 'R^2': statInfo['R2'] = float(linha[2])
            if linha[0] == 'a': statInfo['yInt'] = float(linha[2])
            if linha[0] == 'b': statInfo['slope'] = float(linha[2])
            if linha[0] == 'SE_a': statInfo['yIntErr'] = float(linha[2])
            if linha[0] == 'SE_b': statInfo['slopeErr'] = float(linha[2])
    o_file.close()
    
    E_ref = pd.read_csv(E_ref_file, header=None).to_numpy()
    E_pred = pd.read_csv(E_ML_file, header=None).to_numpy()

    x = np.arange(E_ref.min(), E_ref.max(), 0.001)
    f = lambda x: statInfo['yInt'] + statInfo['slope']*x
    y = np.array([f(_) for _ in x])

    genFig('Correlation between reference and ML Energies',
           '$R^2 = {:.9f}$'.format(statInfo['R2']),
           'Reference energies ($\\mathrm{cm}^{-1}$)',
           'ML energies ($\\mathrm{cm}^{-1}$)')
    plt.plot(E_ref, E_pred, 'k.', label='Data')
    plt.plot(x, y, 'r-', label='Regression: $a = {:.2f}, b = {:.2f}$'.format(statInfo['slope'],statInfo['yInt']))
    
    if fig_name != None: plt.savefig(fig_name)
    if show: plt.show() 
    
def lcPlot(lcPath, graphs, fig_name = None, show = True):
    if 'timePred' in graphs:
        lcTimePred = pd.read_csv('learningCurve/MLatomF/lctimepredict.csv', sep=', ')
        Ntrain = lcTimePred['Ntrain'].to_numpy()
        meanTimePred = lcTimePred['meanTime'].to_numpy()
        sd_timePred = lcTimePred['SD'].to_numpy()
        
        genFig('Learning Curves','Prediction Time',
               'Average time for energy predictions',
               'Number of Training Points')
        plt.errorbar(Ntrain, meanTimePred, yerr=sd_timePred, fmt='g-', label='Prediction Time')
        
    elif 'timeTrain' in graphs:
        lcTimeTrain = pd.read_csv('learningCurve/MLatomF/lctimetrain.csv', sep=', ')
        Ntrain = lcTimeTrain['Ntrain'].to_numpy()
        meanTimeTrain = lcTimeTrain['meanTime'].to_numpy()
        sd_timeTrain = lcTimeTrain['SD'].to_numpy()
        
        genFig('Learning Curves', 'Training Time',
               'Average training time',
               'Number of training points')
        plt.errorbar(Ntrain, meanTime, yerr=sd_timeTrain, fmt='g-', label='Training Time')
        
    elif 'yErr' in graphs:
        lcYerr = pd.read_csv('learningCurve/MLatomF/lcy.csv', sep=', ')
        Ntrain = lcYerr['Ntrain'].to_numpy()
        yErr = lcYerr['meanTime'].to_numpy()
        sd_yErr = lcYerr['SD'].to_numpy()
        
        genFig('Learning Curves', 'Root-Mean-Square Error (RMSE)',
               'RMSE',
               'Number of training points')
        plt.errorbar(Ntrain, yErr, yerr = sd_yErr, fmt = 'g-', label = 'RMSE')
    
    if fig_name != None: plt.savefig(fig_name)
    if show: plt.show() 
        
    
if __name__ == '__main__': #Example run for H2O2-Kr
    #Clears pre-existing data on folder
    os.system('rm -rf *.x *.dat *.out slice* learningCurve/')
    
    #Generates ML input geometries for H2O2-Kr 
    genH2O2_Ng('../ml_scripts/H2O2-Kr.xyz', 'Kr', dR = 0.1, dTeta = 10.)
    
    #Fetches the equilibrium geometry from a pre-made Gaussian 09 log file
    fetchEq('../Logs/H2O2_Kr-opt.log','../ml_scripts/H2O2-Kr_eq.xyz') 
    
    #Fetches the reference energies fromGaussian 09 log files and writes them in MLatom '.dat' files
    E_ref = fetchEnergies('../Logs/MP4/H2O2-Kr','../ml_scripts/H2O2-Kr_E.dat')

    #Converts XYZ molecular geometries into MLatom '.x' input files
    os.system(mlatom_path+'XYZ2X XYZfile=H2O2-Kr.xyz '+ \
                          'XfileOut=x_sorted.dat '+ \
                          'molDescriptor=RE '+ \
                          'molDescrType=sorted '+ \
                          'XYZsortedFileOut=x_sorted.out '+ \
                          'permInvNuclei=1-2.3-4') # specifies which atoms to permute
                              
    #Converts XYZ equilibrium molecular geometry into MLatom X input file
    os.system(mlatom_path+'XYZ2X '+ \
                          'XYZfile=eq.xyz '+ \
                          'XfileOut=eq.x '+ \
                          'molDescriptor=RE '+ \
                          'molDescrType=sorted '+ \
                          'XYZsortedFileOut=eqx_sorted.out '+ \
                          'permInvNuclei=1-2.3-4')
                          
    #slices dataset into 3 equal size regions
    os.system(mlatom_path+'slice nslices=3 XfileIn=x_sorted.dat eqXfileIn=eq.x')
    
    #Samples using structure-based sampling
    os.system(mlatom_path+'sampleFromSlices nslices=3 sampling=structure-based Ntrain=567')
    
    #Merges the sampled indices from all slices
    os.system(mlatom_path+'mergeSlices nslices=3 Ntrain=567')
    
    #Training and estimating accuracy of ML model
    os.system(mlatom_path+'estAccMLmodel '+\
                          'XYZfile=H2O2-Kr.xyz '+\
                          'Yfile=H2O2-Kr_E.dat '+\
                          'YestFile=H2O2-Kr_ML.dat '+\
                          'molDescriptor=RE '+\
                          'molDescrType=permuted '+\
                          'permInvNuclei=1-2.3-4 '+\
                          'kernel=Gaussian '+\
                          'permInvKernel '+\
                          'sigma=opt '+\
                          'lambda=opt '+\
                          'minimizeError=RMSE '+\
                          'sampling=user-defined '+\
                          'itrainin=itrain.dat '+\
                          'itestin=itest.dat '+\
                          'isubtrainin=isubtrain.dat '+\
                          'ivalidatein=ivalidate.dat '+\
                          'Ntrain=567 '+\
                          'Ntest=189 '+\
                          'Nsubtrain=453 > estAcc.out')   
                          
    # Plots the correlation graph between reference and ML energies
    correlFig('../ml_scripts/estAcc.out', '../ml_scripts/H2O2-Kr_E.dat', '../ml_scripts/H2O2-Kr_ML.dat')
    
    #Plots the learning curve
    os.system(mlatom_path+'learningCurve '+\
                          'XYZfile=H2O2-Kr.xyz '+\
                          'Yfile=H2O2-Kr_E.dat '+\
                          'YestFile=H2O2-Kr_ML.dat '+\
                          'lcNtrains=100,216,333,450,567 '+\
                          'lcNrepeats=3 '+\
                          'Nsubtrain=0.8 '+\
                          'Nvalidate=0.2 '+\
                          'Ntest=189 '+\
                          'molDescriptor=RE '+\
                          'molDescrType=permuted '+\
                          'permInvNuclei=1-2.3-4 '+\
                          'kernel=Gaussian '+\
                          'permInvKernel '+\
                          'sigma=opt '+\
                          'lambda=opt > learnCurve.out')  
    
'''
LAWS OF PROGRAMMING DEFINITION:  A WORKING PROGRAM
                                 IS ONE THAT HAS
                                 ONLY UNOBSERVED
                                 BUGS.
'''
