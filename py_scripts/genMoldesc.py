'''
This Python3 script generates molecular descriptors
for MLatom.
'''

import numpy as np
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
                print(Teta)
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
                    print(Teta)
                    i_file.write("{}\n\n".format(M))
                    for i in range(M):
                        i_file.write("{}    {:.5f}  {:.5f}  {:.5f}\n".format(atom[i], x[i], y[i], z[i]))
                        print("{}    {:.5f}  {:.5f}  {:.5f}\n".format(atom[i], x[i], y[i], z[i]))


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
    return c


if __name__ == '__main__':
    genH2O2_Ng('../ml_scripts/H2O2-Kr.xyz', 'Kr', dR = 0.1, dTeta = 10.)
    fetchEq('../../Logs/H2O2_Kr-opt.log','../H2O2-Kr_eq.dat')
#    os.system("mlatom XYZ2X XYZfile=H2O2-Ng.dat XfileOut=x_CM.dat molDescriptor=CM")
