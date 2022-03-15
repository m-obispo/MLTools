import numpy as np

#i_file = sys.argv[1]
#funct = sys.argv[2]
#basis = sys.argv[3]

#try:
#    open(i_file,"r")
#except:
#    print("Erro na abertura do arquivo")
#    exit()


#Distâncias (em angstroms) e ângulos (em graus) da geometria do sistema H2O2-Kr

#with open('teste.xyz','w') as h:
#    for i in range(len(atom)):
#        h.write(cabecalho(ram,nproc,'mp4','aug-cc-PVTz'))
#        h.write(atom[i]+'   '+str(x[i])+'   '+str(y[i])+'   '+str(z[i]))
#        h.write('\n')
#
#t=0
#k=0
#c=0

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
    D = 1.450               #Distância O-O
    d = 0.966               #Distância O-H
    chi = 108.0             #Ângulo O-O-H
    teta1 = 0.0             #Ângulo entre uma das ligações O-H e o eixo y
    teta2 = 0.0             #Ângulo entre uma das ligaçẽos O-H e o eixo y
    
    mram = '8'
    nproc = '8'
    i_file='../Inputs/H2O2-{}/H2O2-{}'.format(Ng, Ng)
    c = 0 

    atom = ['O', 'O', 'H', 'H', Ng]
    M = len(atom)
    dsen = d*np.sin(np.radians(chi))
    dcos = d*np.cos(np.radians(chi))
    D2 = D/2
    
    coords = np.zeros((1,M,3))
    
    def cabecalho(ram, np, Ng, i, opt = False):
        head = "%mem={}GB\n%nproc={}".format(ram, np)
        head += "\n%Chk=/home/matheus/.masters/chk/H2O2-{}_{}.chk".format(Ng, i)
        if opt: 
            head += "\n#p mp4/aug-cc-pvtz int=ultrafine counterpoise=2 opt\n\nH2O2-{}\n\n0,1 0,1 0,1\n".format(Ng)
        else: 
            head += "\n#p mp4/aug-cc-pvtz int=ultrafine counterpoise=2 \n\nH2O2-{}\n\n0,1 0,1 0,1\n".format(Ng)
        return head
    
    r_range = np.arange(3., 5.+dR, dR)
    
    if dAlpha == 0.:
        alpha_range = [90.]
    else:
        alpha_range = np.arange(-90., 90. + dAlpha, dAlpha)
    
    if dTeta == 0.:
        teta_range = [110.]
    else:
        teta_range = np.arange(0., 360., dTeta)
    
    for Alpha in alpha_range:
        cos_a = np.cos(np.radians(Alpha))
        sen_a = np.sin(np.radians(Alpha))
        
        alpha_index = '_{}'.format(np.where(alpha_range == Alpha)[0][0])
        
        for Teta in teta_range:
            teta1 = -Teta/2
            teta2 = Teta/2
            cos_t1 = np.cos(np.radians(teta1))
            cos_t2 = np.cos(np.radians(teta2))
            sen_t1 = np.sin(np.radians(teta1))
            sen_t2 = np.sin(np.radians(teta2))
            
            teta_index ='_{}'.format(np.where(teta_range == Teta)[0][0])
            
            for R in r_range:
                r_index = '_{}'.format(np.where(r_range == R)[0][0])
                i_file += alpha_index + teta_index + r_index 
                
                #              O   O   H            H            Ng
                x = np.array([[0., 0., dsen*sen_t1, dsen*sen_t2, 0.     ]])
                y = np.array([[0., 0., dsen*cos_t1, dsen*cos_t2, R*cos_a]])
                z = np.array([[D2, -D2, D2 - dcos, - D/2 + dcos, R*sen_a]])
                xyz = np.append(np.append(x.T, y.T, axis=1), z.T, axis=1).reshape(1,M,3)
                coords = np.append(coords, xyz, axis=0)
                
                if Alpha == 90. and Teta == 110.:
                    with open('../Inputs/H2O2-{}/H2O2-{}_opt.com'.format(Ng, Ng),'w') as h:
                        h.write(cabecalho(mram, nproc, Ng, 'opt', opt=True))
                        for i in range(len(atom)-1):
                            h.write("{}(Fragment=1)    {:.5f}  {:.5f}  {:.5f}\n".format(atom[i], 
                                                                                        x[0,i], 
                                                                                        y[0,i], 
                                                                                        z[0,i]))
                        h.write('{}(Fragment=2)   0.    3.5    0.'.format(Ng))
                        h.write("\n\n")
                        
                    with open('../Inputs/H2O2-{}/H2O2-{}_inf.com'.format(Ng, Ng),'w') as h:
                        h.write(cabecalho(mram, nproc, Ng, 'inf'))
                        for i in range(len(atom)-1):
                            h.write("{}(Fragment=1)    {:.5f}  {:.5f}  {:.5f}\n".format(atom[i], 
                                                                                        x[0,i], 
                                                                                        y[0,i], 
                                                                                        z[0,i]))
                        h.write('{}(Fragment=2)   0.    15.    0.'.format(Ng))
                        h.write("\n\n")
                
                with open(i_file+'.com','w') as h:
                    # print(alpha_index, teta_index, r_index)
                    # print(i_file)
                    h.write(cabecalho(mram, nproc, Ng, i_file[-5:]))
                    for i in range(len(atom)-1):
                        h.write("{}(Fragment=1)    {:.5f}  {:.5f}  {:.5f}\n".format(atom[i], 
                                                                                    x[0,i], 
                                                                                    y[0,i], 
                                                                                    z[0,i]))
                    h.write('{}(Fragment=2)   0.    {:.5f}    {:.5f}'.format(Ng, y[0,-1], z[0,-1]))
                    h.write("\n\n")
                    
                    c += 1
                    perc = c/len(alpha_range)/len(teta_range)/len(r_range)
                    print('Progress: [{}{}] {:.1f} %'.format('#'*int(np.floor(perc*20)),'-'*int(np.floor((1-perc)*20)),
                                                         perc*100), end='\r')
                    # print('Progress:[{}{}] {} %'.format())
                    
                i_file='../Inputs/H2O2-{}/H2O2-{}'.format(Ng, Ng)

    return atom, coords[1:]

                 
if __name__ == '__main__':
    for Ng in ['Kr']: #['He', 'Ne', 'Ar', 'Kr', 'Xe']
        atom, coords = genH2O2_Ng(Ng, dR=0.1, dTeta=10., dAlpha=10.)
        # print(atom, '\n', coords)

    #Pai, te amo

#0 1
#O
#O       1        1.45000
#H       1        0.96600     2      108.00000
#H       2        0.96600     1      108.00000     3        0.02562
#Kr      3        4.20766     1      147.92161     2        0.02562
