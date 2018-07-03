''' Fetching shit, this takes a lot of code... '''




import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *
import sys
import os
import spacepy

#from initiate 
import variables as var




class fetch():
    def __init__(self):
        #self.all_data = ['self.date', 'self.t', 'Bx', 'By', 'Bz', 'IGRFx', 'IGRFy', 'IGRFz']
        #self.dataname = var.dataname 
        #os.mkdir('graphs')
        if not os.path.exists('graphs'):
            os.makedirs('graphs')
        return None
        
    def help(self):
        print '''Functions:
        fetch() fetches filename(variables.py).asc file and makes an npy file.
        ....
        '''
        
    def olddata(self): #, filename):
        '''Opens filename (*.asc) and saves it in an npy file.
        should be made more general.
        date, t, Bx, By, Bz, IGRFx, IGRFy, IGRFz '''
        files = open(var.filename_old, 'r') #'ICI4_magn_hires.asc', 'r')
        asc_len = True
        
        date = [] ; t = []
        Bx = [] ; By = [] ; Bz = []
        IGRFx = [] ; IGRFy = [] ; IGRFz = []
        
        # Read header info
        for i in range(var.header_length_old):
            print files.readline()

        # Read contents
        dictlab = files.readline()
        content = dictlab.split()
        
        #This seriously needs an update:
        f = 0
        while asc_len == True:
            Datt = files.readline()
            ny = Datt.split()
            Date, T, BX, BY, BZ, a1, a2, a3, a4, a5, a6, IGRFX, IGRFY, IGRFZ = ny
            
            date.append(float(Date)) ; t.append(float(T))
            Bx.append(float(BX)) ; By.append(float(BY)) ; Bz.append(float(BZ))
            IGRFx.append(float(IGRFX))
            IGRFy.append(float(IGRFX))
            IGRFz.append(float(IGRFX))
            f +=1
            if f> 1728058:
                asc_len = False
        print ' Read from file'
        
        proj_data = np.array(date), np.array(t), np.array(Bx), np.array(By), np.array(Bz), np.array(IGRFx), np.array(IGRFy), np.array(IGRFz)
        np.save(var.dataname_old, proj_data)
        print ' Saved in file', var.dataname_old
        return None 
        

    def B(self):
        '''Opens filename (*.asc) and saves it in an npy file.
        should be made more general.'''
        files = open(var.filename_B, 'r')
        asc_len = True
        
        t = [] ; t_num = []
        Bx = [] ; By = [] ; Bz = []
        
        # Read header info
        for i in range(var.header_length_B):
            files.readline()
        # Read contents
        dictlab = files.readline()
        self.content = dictlab.split()

        f = 0
        while asc_len == True:
            Datt = files.readline()
            ny = Datt.split()
            if len(ny) == 3:
                Bx.append(ny[0]) ; By.append(ny[1]) ; Bz.append(ny[2])
            elif len(ny) == 4:
                t.append(ny[0]) ; t_num.append(int(f))
                Bx.append(float(ny[1])) ; By.append(float(ny[2])) ; Bz.append(float(ny[3]))
            else:
                asc_len = False
            f +=1
        # NEXT LINES OF CODE ARE UGLY AF PLS IGNORE MUST FIX LATER 
        # LINEAR FIT FOR TIME, IS ALMOST PERFECT THO
        #translate T!!!1 + fill out (linearly)
        tt = spacepy.time.Ticktock(t)
        print tt
        
        #tt = tt.getUNX
        t = tt.TAI
        print t
        t -= t[0] 
        t_abs = tt.TAI #IMPORTANTERER
        t_abs = np.linspace(t_abs[0], t_abs[-1], len(Bx))
        T = np.zeros(len(Bx))
        #for j in range(len(t)):
        #    T[t_num[j]] = t[j]
        minn = int(0.08*len(Bx))
        #T = np.linspace(T[0], np.max(T[-minn-100:-minn]), len(Bx)-minn)
        T = np.linspace(T[0], np.max(t[-100: ]), len(Bx), dtype=float)
        print ' Read from file ', var.filename_B
        
        proj_data = t_abs, T, np.array(Bx, dtype=float), np.array(By, dtype=float), np.array(Bz, dtype=float)
        np.save(var.dataname_B, proj_data)
        print ' Saved in file ', var.dataname_B
        return None 
        

    def position(self):
        '''Opens filename (20151022ICI4.dat) and saves it in an npy file.
        should be made more general.'''
        files = open(var.filename_pos, 'r')
        asc_len = True
        
        t = []
        lat = [] ; lon = [] ; alt = []
        theta_usr = [] ; phi_usr = []
        
        # Read header info
        for i in range(var.header_length_pos):
            files.readline()
        # Read contents
        dictlab = files.readline()
        content = dictlab.split()
        # Theta_E Phi_E Theta_E Phi_E Latitude Longitude Alt [m] 
        # TM-X axis    TM-Y axis    TM-Z axis  phase [deg]
        f = 0
        while asc_len == True:
            Datt = files.readline()
            ny = Datt.split()
            if len(ny) > 4:
                t.append(float(ny[0]))
                theta_usr.append(float(ny[1])) ; phi_usr.append(float(ny[2]))
                lat.append(float(ny[5])) ; lon.append(float(ny[6])) ; alt.append(float(ny[7]))
            else:
                asc_len = False
            f +=1
            
        # NEXT LINES OF CODE ARE UGLY AF PLS IGNORE MUST FIX LATER 
        #translate T!!!1 + fill out (linearly)

        print ' Read from file ', var.filename_pos
        proj_data = np.array(t), np.array(theta_usr), np.array(phi_usr), np.array(lat), np.array(lon), np.array(alt)*1e-3, 
        np.save(var.dataname_pos, proj_data)
        print ' Saved in file ', var.dataname_pos
        return None 




