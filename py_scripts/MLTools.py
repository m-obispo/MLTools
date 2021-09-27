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
    global keywords_mpn
    e_file = open(E_file_name,'w')
    MP4 = np.zeros(21)
    c = 0
    for k in range(36): #[0,17]: 
        mp4 = []
        with open(i_file_name+"_{}.log".format(k),'r') as i_file:
            for line in i_file:
                linha = line.split()
                if linha[:3] == keywords_mpn:
                    mp4.append(float(linha[-1]))
                    c += 1
                    print('Energy no. {}: {:.5f} Ha'.format(c,float(linha[-1])))
                    e_file.write('{:.5f}\n'.format(float(linha[-1])))
        MP4 = np.vstack((MP4,np.array(mp4)))
    MP4 = MP4[1:,:]
    e_file.close()
    return MP4

def genFig(xLabel, yLabel, Title = '', subTitle = ''):
    fig = plt.figure(figsize = (10, 10), dpi=170)
    plt.suptitle(Title)
    plt.title(subTitle)
    plt.grid()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks()
    plt.yticks()

def correlFig(out_file_name, E_ref_file, E_ML_file, fig_name = None, show = True):
    o_file = open(out_file_name, 'r')
    c = 0
    statInfo = {}
    for line in o_file:
        linha = line.split()
        if linha == ["CREATE","AND","SAVE","FINAL","ML","MODEL"]: c += 1
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

    genFig('Reference energies (Ha)',
           'ML energies (Ha)',
           'Correlation between reference and ML Energies',
           '$R^2 = {:.5f}$'.format(statInfo['R2'])
    )
    plt.plot(E_ref, E_pred, 'k.', label='Data')
    plt.plot(x, y, 'r-', label='Regression: $a = {:.2f}, b = {:.2f}$'.format(statInfo['slope'],statInfo['yInt']))
    if fig_name != None: plt.savefig(fig_name)
    if show: plt.show()

#if __name__ == '__main__':
#    genH2O2_Ng('../ml_scripts/H2O2-Kr.xyz', 'Kr', dR = 0.1, dTeta = 10.)
#    fetchEq('../Logs/H2O2_Kr-opt.log','../ml_scripts/H2O2-Kr_eq.xyz')
#    fetchEnergies('../Logs/MP4/H2O2-Kr','../ml_scripts/H2O2-Kr_energies.dat')
#    os.system("mlatom XYZ2X XYZfile=H2O2-Ng.dat XfileOut=x_CM.dat molDescriptor=CM")

'''
LAWS OF PROGRAMMING DEFINITION:  A WORKING PROGRAM
                                 IS ONE THAT HAS
                                 ONLY UNOBSERVED
                                 BUGS.
'''
