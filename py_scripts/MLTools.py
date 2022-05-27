'''
This Python3 script generates molecular descriptors
for MLatom.
'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import os
from ase import Atoms
from ase.calculators.gaussian import Gaussian, GaussianOptimizer

from config import *

def genH2O2_Ng(Ng, dR, dTeta=0., dAlpha=0.):
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
       
    atom = ['O', 'O', 'H', 'H', Ng]
    nAtoms = len(atom)						#No of atoms (nuclei)
    coords = np.zeros((1,nAtoms,3))
    
    dsen = d*np.sin(np.radians(chi))
    dcos = d*np.cos(np.radians(chi))
    D2 = D/2
    c = 0 
       
    if dAlpha == 0.:
        alpha_range = [90.]
    else:
        alpha_range = np.arange(-90., 0. + dAlpha, dAlpha)
    
    if dTeta == 0.:
        teta_range = [110.]
    else:
        teta_range = np.arange(0., 360., dTeta)
    
    for Alpha in alpha_range:
        cos_a = np.cos(np.radians(Alpha))
        sen_a = np.sin(np.radians(Alpha))
        for Teta in teta_range:
            teta1 = -Teta/2
            teta2 = Teta/2
            cos_t1 = np.cos(np.radians(teta1))
            cos_t2 = np.cos(np.radians(teta2))
            sen_t1 = np.sin(np.radians(teta1))
            sen_t2 = np.sin(np.radians(teta2))
            r_range = np.arange(3., 5.+dR, dR)
            for R in r_range:
                #              O    O   H            H            Ng
                x = np.array([[0.,  0., dsen*sen_t1, dsen*sen_t2, 0.     ]]).T
                y = np.array([[0.,  0., dsen*cos_t1, dsen*cos_t2, R*cos_a]]).T
                z = np.array([[D2, -D2, D2 - dcos  , -D2 + dcos , R*sen_a]]).T
                xyz = np.append(np.append(x, y, axis=1), z, axis=1).reshape(1,nAtoms,3)
                coords = np.append(coords, xyz, axis=0)
                
                c += 1
                perc = c/len(alpha_range)/len(teta_range)/len(r_range)
                print('Progress: [{}{}] {:.1f} %'.format('#'*int(np.floor(perc*20)),
                                                         '-'*int(np.floor((1-perc)*20)),
                                                         perc*100), end='\r')
                
    return atom, coords[1:]
        

def genMLatomInput(i_file_name, atom, coords):
    '''
    Generates MLatom-compatible .xyz files for arbitrary molecular systems.
    --------------------------------------------------------------------------------
    Params:
      atom (arr, str): List of atoms in the molecule.
      coords (arr): A NxMx3 array corresponding to the molecular geometries at each 
                    step of the calculation.
    '''
    M = len(atom)
    c = 0 
    with open(i_file_name, 'w') as i_file:
        for j in range(coords.shape[0]):
            i_file.write("{}\n\n".format(M))
            x = coords[j,:,0]
            y = coords[j,:,1]
            z = coords[j,:,2]
            for i in range(M):
                i_file.write("{}    {:.5f}  {:.5f}  {:.5f}\n".format(atom[i], x[i], y[i], z[i]))

                c += 1
                perc = c/coords.shape[0]
                print('Progress: [{}{}] {:.1f} %'.format('#'*int(np.floor(perc*20)),
                                                         '-'*int(np.floor((1-perc)*20)),
                                                         perc*100), end='\r')

def genGaussianInputs(atoms, coords, output_dir, index=None):
    def cabecalho(ram, np, Ng, i, opt = False):
        head = "%mem={}GB\n%nproc={}".format(ram, np)
        head += "\n%Chk=/home/matheus/.masters/chk/H2O2-{}_{}.chk".format(Ng, i)
        if opt: 
            head += "\n#p mp4/aug-cc-pvtz int=ultrafine counterpoise=2 opt\n\nH2O2-{}\n\n0,1 0,1 0,1\n".format(Ng)
        else: 
            head += "\n#p mp4/aug-cc-pvtz int=ultrafine counterpoise=2 \n\nH2O2-{}\n\n0,1 0,1 0,1\n".format(Ng)
        return head
    
    c = 0 
    Ng = atoms[-1]

    if str(type(index)) == "<class 'NoneType'>":
        index = range(coords.shape[0])
    
    for i in index:
        x = coords[i,:,0]
        y = coords[i,:,1]
        z = coords[i,:,2]
        with open(output_dir+'/H2O2-{}_{:.0f}.com'.format(Ng,i),'w') as h:
            h.write(cabecalho(nproc,ram, Ng, i))
            for j in range(len(atoms)-1):
                h.write("{}(Fragment=1)    {:.5f}  {:.5f}  {:.5f}\n".format(atoms[j], x[j], y[j], z[j]))
            h.write("{}(Fragment=2)    {:.5f}  {:.5f}  {:.5f}\n".format(atoms[-1], x[-1], y[-1], z[-1]))
            h.write("\n\n")

            c += 1
            perc = c/len(index)
            print('Progress: [{}{}] {:.1f} %'.format('#'*int(np.floor(perc*20)),
                                                     '-'*int(np.floor((1-perc)*20)),
                                                     perc*100), end='\r')

def refE_gaussian(atom, coords):
    '''
    Performs single-point energy calculations for arbitrary molecular systems using
    Gaussian in the backend.
    --------------------------------------------------------------------------------
    Params:
      atom (arr, str): List of atoms in the molecule.
      coords (arr): A NxMx3 array corresponding to the molecular geometries at each 
                    step of the calculation.
    '''
    E = np.array([])
    for i in range(coords.shape[0]):
        molSys = Atoms(atom, coords[i])
        molSys.calc = Gaussian(nprocshared='16',
                               mem='16GB',
                               chk='MyJob.chk',
                               save=None,
                               method='mp4',
                               basis='aug-cc-pvtz',
                               charge=0,
                               mult=1)
        E = np.append(E, molSys.get_potential_energy())
    return E

def dfMaker(atoms, coords):
    indices_list = list(itertools.product(np.arange(coords.shape[0]),atoms))
    indxs = pd.MultiIndex.from_tuples(indices_list, names=['i', 'atoms'])
    coords_2D = coords.reshape(coords.shape[0]*coords.shape[1], coords.shape[2]) 
    geoms = pd.DataFrame(coords_2D, index, ['x', 'y', 'z'])
    return geoms

def SBS_to_Gaussian(atoms, coords, output_dir,itrain_file='../ml_scripts/itrain.dat'):
    itrain = pd.read_csv(itrain_file,header=None,names=['itrain']).to_numpy()
    itrain = itrain.reshape((itrain.shape[0],)) 
    itrain.sort()
    genGaussianInputs(atoms,coords,output_dir,itrain-1)
    print('\nReady')

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

    
def fetchEnergy(i_file_name):
    '''
    Generates MLatom-compatible reference energy values from Gaussian logs for
    arbitrary molecular systems. Currently, only MPn/aug-cc-pVTZ level calculations
    are supported.
    --------------------------------------------------------------------------------
    Params:
      i_file_name (str): Name of .log file to be parsed WITH EXTENSION.
      E_file_name (str): Name of .dat file to be generated with reference energies.
    '''
    
    with open(i_file_name,'r') as i_file:
        l = 0
        for line in i_file:
            linha = line.split()
            # print(l, linha)
            if len(linha) > 0 and linha[-1] == '0=g16':
                keywords_phf = ['Counterpoise', 'corrected', 'energy']
                l += 1
            elif len(linha) > 0 and linha[-1] == '0=g09':
                keywords_phf = ['Counterpoise:', 'corrected', 'energy']
                l += 1
            elif l > 0 and linha[:3] == keywords_phf:
                mp4 = float(linha[-1])
                l += 1
            elif l < 2:
                mp4 = '0.0'
                # print('Energy no. {}: {:.5f} Ha'.format(c,float(linha[-1])))
                
    return mp4

def fetchEnergies(i_file_name, E_file_name, scan=False, verbose=False):
    '''
    Generates MLatom-compatible reference energy values from Gaussian logs for
    arbitrary molecular systems. Currently, only MPn/aug-cc-pVTZ level calculations
    are supported.
    --------------------------------------------------------------------------------
    Params:
      i_file_name (str): Name of .log file to be parsed WITHOUT EXTENSION.
      E_file_name (str): Name of .dat file to be generated with reference energies.
            scan (bool): If the 'Scan' feature of Gaussian was used.
         verbose (bool): Sets whether to display more info or not
    '''
    
    e_file = open(E_file_name,'w')
    c = 0
        
    Einf = fetchEnergy(i_file_name+"_inf.log")
    
    if scan:
        MP4 = np.zeros(21)
        for k in range(36):
            mp4 = []
            with open(i_file_name+"_{}.log".format(k),'r') as i_file:
                l = 0
                for line in i_file:
                    linha = line.split()
                    if len(linha) > 0 and linha[-1] == '0=g16':
                        keywords_phf = ['Counterpoise', 'corrected', 'energy']
                    elif len(linha) > 0 and linha[-1] == '0=g09':
                        keywords_phf = ['Counterpoise:', 'corrected', 'energy']
                    if linha[:3] == keywords_phf:
                        mp4.append(float(linha[-1]))
                        e_file.write('{:.5f}\n'.format((float(linha[-1]) - Einf)))
                        c += 1
                        # print('Energy no. {}: {:.5f} Ha'.format(c,float(linha[-1])))
            MP4 = np.vstack((MP4,np.array(mp4)))
    else:
        MP4 = np.zeros((4,36,21))
        for i in range(0,10,3):
            for j in range(36):
                for k in range(21):
                    E = fetchEnergy(i_file_name+"_{}_{}_{}.log".format(i,j,k))
                    MP4[i//3,j,k] = float(E) - Einf
                    if verbose: print('Energy no. {} {} {}: {} cm⁻¹'.format(i,j,k,MP4[i//3,j,k]*219474.6))
                    e_file.write('{:.5f}\n'.format((float(E) - Einf)))
        
    e_file.close()
    return MP4

def genFig(Title, subTitle = '', xLabel='', yLabel=''):
    '''
    Autogenerates figure configurations.
    --------------------------------------------------------------------------------
    Params:
      Title (str): The main title of the figure.
      subTitle (str): The subtitle of the figure.
      xLabel (str): The x-axis label.
      yLabel (str): The y-axis label.
    '''
    fig = plt.figure(figsize = (10, 10), dpi=200)
    plt.suptitle(Title)
    plt.title(subTitle)
    plt.grid()
    plt.xlabel(xLabel)
    plt.ylabel(yLabel)
    plt.xticks()
    plt.yticks()

def correlFig(out_file_name, E_ref_file, E_ML_file, itest_file, fig_name = None, show = True):
    '''
    Generates a correlation plot between reference and ML energies
    from the .dat files and MLatom output.
    --------------------------------------------------------------------------------
    Params:
      out_file_name (str): Name of MLatom output file containing statistical analysis.
      E_ref_file (str): Name of .dat file containing reference energies.
      E_ML_file (str): Name of .dat file containing ML energies.
      fig_name (str): Name of the figure file to be saved. If set to 'None' no image 
                      file will be saved. Defaults to 'None'
      show (bool): Whether the plot will be displayed. Defaults to 'True'.
    '''
    global keywords_estAcc, statInfo, E_ref, E_pred, E_ref_test, E_pred_test
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
    
    E_ref.reshape((len(E_ref),))
    E_pred.reshape((len(E_pred),))
    
    itest = pd.read_csv(itest_file, header=None).to_numpy()
    itest.reshape((len(itest),))
    
    E_ref_test = np.array([E_ref[i-1] for i in itest])
    E_pred_test = np.array([E_pred[i-1] for i in itest])
    
    E_ref_test = E_ref_test.reshape((len(E_ref_test),))
    E_pred_test = E_pred_test.reshape((len(E_pred_test),))    

    x_plot = np.arange(E_ref_test.min(), E_ref_test.max(), 0.001)
    # x = np.arange(E_ref_.min(), E_ref_t.max(), 0.001)
    f = lambda x: statInfo['yInt'] + statInfo['slope']*x
    y_plot = np.array([f(x) for x in x_plot])

    genFig('Correlation between reference and ML Energies',
           'MSE = {:.5e}'.format(statInfo['MSE']),
           'Reference energies ($\\mathrm{cm}^{-1}$)',
           'ML energies ($\\mathrm{cm}^{-1}$)')
    # plt.plot(E_ref, E_pred, 'k.', label='Data')
    plt.plot(E_ref_test, E_pred_test, 'k.', label='Data')
    plt.plot(x_plot, y_plot, 'r-', label='Regression: $a = {:.2f}, b = {:.2f}$'.format(statInfo['slope'],statInfo['yInt']))
    plt.legend()
    if fig_name != None: plt.savefig(fig_name)
    if show: plt.show() 

def pesPlot(E_file, Rmin=3., Rmax=5., dR=0.1, Tmin=0., Tmax=360., dTeta=10., Amin=0., Amax=90., dAlpha=30.):
    Teta,Alpha,R = np.meshgrid(np.arange(Tmin, Tmax,10), np.arange(0, 91, 30), np.arange(3,5.1,0.1))
    E = pd.read_csv(E_file,header=None).to_numpy().reshape((4,36,21))
    
    fig = plt.figure(figsize = (12, 12), dpi = 175)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(30, 45)
    ax.w_xaxis.pane.fill = False
    ax.w_yaxis.pane.fill = False
    ax.w_zaxis.pane.fill = False
    ax.grid(False) 

    kw = {
        'vmin': E.min(),
        'vmax': E.max(),
        'levels': np.linspace(E.min(), E.max(), 20),
    }

    #AlphaxTheta
    ax.contourf(E[:,:,0], Teta[:,:,0], Alpha[:,:,0], zdir='x', cmap='Greens_r', offset=3, **kw)
    ax.contour(E[:,:,0], Teta[:,:,0], Alpha[:,:,0], zdir='x', cmap='Greys_r', offset=3, **kw)

    #AlphaxR
    ax.contourf(R[:,0,:], E[:,0,:], Alpha[:,0,:], zdir='y', cmap='Greens_r', offset=0, **kw)
    ax.contour(R[:,0,:], E[:,0,:], Alpha[:,0,:], zdir='y', cmap='Greys_r', offset=0, **kw)

    #RxTheta
    im = ax.contourf(R[0,:,:], Teta[0,:,:], E[0,:,:], zdir='z', cmap='Greens_r', offset=0, **kw)
    ax.contour(R[0,:,:], Teta[0,:,:], E[0,:,:], zdir='z', cmap='Greys_r', offset=0, **kw)
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.05, format='%.3f', label='E $(cm^{-1})$')

    Rmin, Rmax = R.min(), R.max()
    ax.set(xlim=[Rmin, Rmax], xlabel='$R  (\\mathring{A})$', ylabel='$\\theta (°)$', zlabel='$\\alpha (°)$')
    plt.suptitle('Superfície de Energia Potencial', fontsize=20)
    plt.show()

    # ax.fill_between(x, y_est - y_err, y_est + y_err, alpha=0.2)
def lcPlot(lcPath, graphs, fig_name = None, show = True):
    '''
    Plots the generated learning curves from MLatom's learningCurve feature. 
    Currently, only one learning curve at a time is available.
    --------------------------------------------------------------------------------
    Params:
      lcPath (str): Name of MLatom's learningCurve directory containing statistical 
                    analysis of the learning curves.
      graphs (array): keywords for different plots.
        'timePred': Average time for energy predictions vs Number of Training Points.
        'timeTrain': Average training time vs Number of Training Points.
        'yErr': Root-mean-square error vs Number of Training Points.
      E_ML_file (str): Name of .dat file containing ML energies.
      fig_name (str): Name of the figure file to be saved. If set to 'None' no image 
                      file will be saved. Defaults to 'None'
      show (bool): Whether the plot will be displayed. Defaults to 'True'.
    '''
    if 'timePred' in graphs:
        lcTimePred = pd.read_csv(lcPath+'lctimepredict.csv', sep=', ', engine='python')
        genFig('Learning Curves','Prediction Time',
               'Number of Training Points', 'Average time for energy predictions (s)')
        plt.plot(lcTimePred['Ntrain'], lcTimePred['meanTime'], 'g.-', label='Prediction Time')
        plt.fill_between(lcTimePred['Ntrain'], lcTimePred['meanTime'] - lcTimePred['SD'],
                         lcTimePred['meanTime'] + lcTimePred['SD'], color = 'g', alpha=0.2)
        
    elif 'timeTrain' in graphs:
        lcTimeTrain = pd.read_csv(lcPath+'lctimetrain.csv', sep=', ', engine='python')
        genFig('Learning Curves', 'Training Time',
               'Number of training points', 'Average training time (s)')
        plt.plot(lcTimeTrain['Ntrain'], lcTimeTrain['meanTime'], 'b.-', label='Training Time')
        plt.fill_between(lcTimeTrain['Ntrain'], lcTimeTrain['meanTime'] - lcTimeTrain['SD'],
                         lcTimeTrain['meanTime'] + lcTimeTrain['SD'], color = 'b', alpha=0.2)
        
    elif 'yErr' in graphs:
        lcYerr = pd.read_csv(lcPath+'lcy.csv', sep=', ', engine='python')
        genFig('Learning Curves', 'Root-Mean-Square Error (RMSE)',
               'Number of training points', 'RMSE (cm$^{-1}$)')
        # plt.errorbar(lcYerr['Ntrain'], lcYerr['meanRMSE'], 
                     # yerr = lcYerr['SD'], fmt = 'r.-', label = 'RMSE')
        plt.plot(lcYerr['Ntrain'], lcYerr['meanRMSE'], 'r.-', label = 'RMSE')
        plt.fill_between(lcYerr['Ntrain'], lcYerr['meanRMSE'] - lcYerr['SD'],
                         lcYerr['meanRMSE'] + lcYerr['SD'], color = 'r', alpha=0.2)
                         
    plt.legend()
    if fig_name != None: plt.savefig(fig_name)
    if show: plt.show() 

if __name__ == '__main__': #Example run for H2O2-Kr
    #Clears pre-existing data on folder
    #os.system('rm -rf ../ml_scripts/*.x ../ml_scripts/*.dat ~/ml_scripts/*.out ../ml_scripts/slice* ../ml_scripts/learningCurve/')
    
    #Generates ML input geometries for H2O2-Kr 
    atoms, coords = genH2O2_Ng('Kr', dR = 0.1, dTeta = 10.)
    genMLatomInput('../ml_scripts/H2O2-Kr.xyz', atoms, coords)
    
    #Fetches the equilibrium geometry from a pre-made Gaussian 09 log file
    #fetchEq('../Logs/H2O2-Kr_opt.log','../ml_scripts/H2O2-Kr_eq.xyz') 
    
    #Converts XYZ equilibrium molecular geometry into MLatom X input file
    os.system(mlatom_path+'XYZ2X '+ \
                          'XYZfile=../ml_scripts/eq.xyz '+ \
                          'XfileOut=../ml_scripts/eq.x '+ \
                          'molDescriptor=RE '+ \
                          'molDescrType=sorted '+ \
                          'XYZsortedFileOut=../ml_scripts/eqx_sorted.out '+ \
                          'permInvNuclei=1-2.3-4')
    
    #Converts XYZ molecular geometries into MLatom '.x' input files
    os.system(mlatom_path+'XYZ2X XYZfile=../ml_scripts/H2O2-Kr.xyz '+ \
                          'XfileOut=../ml_scripts/x_sorted.dat '+ \
                          'molDescriptor=RE '+ \
                          'molDescrType=sorted '+ \
                          'XYZsortedFileOut=../ml_scripts/x_sorted.out '+ \
                          'permInvNuclei=1-2.3-4') # specifies which atoms to permute
                          
    #slices dataset into 3 equal size regions
    os.system(mlatom_path+'slice nslices=3 XfileIn=../ml_scripts/x_sorted.dat eqXfileIn=../ml_scripts/eq.x')
    
    #Samples using structure-based sampling
    os.system(mlatom_path+'sampleFromSlices nslices=3 sampling=structure-based Ntrain=567')
    
    #Merges the sampled indices from all slices
    #os.system(mlatom_path+'mergeSlices nslices=3 Ntrain=567')
    os.system(mlatom_path+'mergeSlices nslices=3 Ntrain=2268')
    
    #Fetches the reference energies from Gaussian 09 log files and writes them in MLatom-compatible '.dat' files
    E_ref = fetchEnergies('../Logs/MP4/H2O2-Kr','../ml_scripts/H2O2-Kr_E.dat')
    
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
                          'Ntrain=2268 '+\
                          'Ntest=756 '+\
                          'Nsubtrain=567 > estAcc.out')   
                          
    # Plots the correlation graph between reference and ML energies
    correlFig('../ml_scripts/estAcc.out', '../ml_scripts/H2O2-Kr_E.dat', 
              '../ml_scripts/H2O2-Kr_ML.dat', '../ml_scripts/itest.dat')
    
    #Plots the learning curve
    os.system(mlatom_path+'learningCurve '+\
                          'XYZfile=H2O2-Kr.xyz '+\
                          'Yfile=H2O2-Kr_E.dat '+\
                          'YestFile=H2O2-Kr_ML.dat '+\
                          'lcNtrains={} '.format(','.join([str(int(x)) for x in np.floor(np.linspace(567,2268,10))]))+\
                          'lcNrepeats=10 '+\
                          'Nsubtrain=0.75 '+\
                          'Nvalidate=0.25 '+\
                          'Ntest=756 '+\
                          'molDescriptor=RE '+\
                          'molDescrType=permuted '+\
                          'permInvNuclei=1-2.3-4 '+\
                          'kernel=Gaussian '+\
                          'permInvKernel '+\
                          'sigma=opt '+\
                          'lambda=opt > learnCurve.out')  
                          #0.8 0.2 189
    #','.join([str(int(x)) for x in np.floor(np.linspace(100,567,10))])
    #.format(','.join([str(x) for x in range(10,0,-1)]))
    # lcPlot('learningCurve/MLatomF/', ['yErr'])
    # lcPlot('learningCurve/MLatomF/', ['timeTrain'])
    # lcPlot('learningCurve/MLatomF/', ['timePred'])
    
'''
LAWS OF PROGRAMMING DEFINITION:  A WORKING PROGRAM
                                 IS ONE THAT HAS
                                 ONLY UNOBSERVED
                                 BUGS.
'''
