'''
Configuration file for MLTools.
'''

# Path to MLatom executable
mlatom_path = 'mlatom '

# System Studied (restricted to H2O2-Ng variants atm)
molsys = 'H2O2-Ng'

# Bond distances (in angstroms) and angles (in degrees) of H2O2-Ng system geometry
D = 1.450                         #O-O distance
d = 0.966                         #O-H distance
chi = 108.0                       #O-O-H angle
teta1 = 0.                        #Angle between one of the O-H bond and y axis
teta2 = 0.                        #Angle between the other O-H bond and y axis

#Keyword lists for Gaussian log parsing
keywords_opt = ['OPTIMIZATION STOPPED','Z-Matrix','---------------------------------------------------------------------']

nproc='8'
ram='8'

#Keyword listing for MLatom
keywords_estAcc = ['test','set']

#Periodic Dictionary of Elements
Elements = {
   'H' : 1,
   'He': 2,
   'Li': 3,
   'Be': 4,
   'B' : 5,
   'C' : 6,
   'N' : 7,
   'O' : 8,
   'Ne': 10,
   'Ar': 18,
   'Kr': 36,
   'Xe': 54,
}
