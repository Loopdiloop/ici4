

import numpy as np
import matplotlib.pyplot as plt
#import spacepy as space
import constants as con
import var_run as run




class reverse_ici():
    def __init__(self):
        
        return None
    
    def init_magnetic(self):
        #Magn. field. Cartesian
        Bx = raw_input('Bx = ')
        By = raw_input('By = ')
        Bz = raw_input('Bz = ')
        try:
            Bx = float(Bx) ; By = float(By) ; Bz = float(Bz)
        except:
            print 'Ran default magnetic field'
            Bx = 0.05 ;  By = 0.006 ; Bz = 0.
        print Bx
        self.B = np.array([Bx, By, Bz])        
        self.absB = abs(self.B)
        print self.B

    def path(self):
        
        chromer_update()
        dump_to_file()


    
    def gravity(self):
        self.g = con.G*run.m_rocket*con.m_earth/self.r**2
        
    def chromer_update():
        self.a = self.g


class make_path():
    def __init__(self):
        self.x = 0.
        self.y = 0.
        self.z = 3e3 #m
        
        self.vx = 5. #m/s
        self.vy = 7. #m/s
        self.vz = 2. #m/s
        return None

    def run():
        return None
        
        
        
        





run = reverse_ici()
run.init_magnetic()

https://spaceflightsystems.grc.nasa.gov/education/rocket/Images/rktaero.gif
