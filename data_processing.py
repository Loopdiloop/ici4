
# Launch in 2015

# Boom fully deployed at 70.00000 s
# Data began ca. 10 sek before launch
# 


import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *
import sys
import os
import spacepy
from spacepy import coordinates as coord
import spacepy.datamodel as spacedatamodel

#from initiate 
import variables as var




class generate():
    def __init__(self):
        #self.all_data = ['self.date', 'self.t', 'Bx', 'By', 'Bz', 'IGRFx', 'IGRFy', 'IGRFz']
        #self.dataname = var.dataname 
        #os.mkdir('graphs')
        self.range_set_done = False
        if not os.path.exists('graphs'):
            os.makedirs('graphs')
        return None
        
    def help(self):
        print '''Functions:
        fetch() fetches filename(variables.py).asc file and makes an npy file.
        ....
        '''
        

    def load(self): #, name):
        self.t_abs, self.t, self.Bx, self.By, self.Bz = np.load(var.dataname_B+'.npy')
        self.t_pos, self.theta_usr, self.phi_usr, self.lat, self.lon, self.alt = np.load(str(var.dataname_pos)+'.npy')
        self.B = np.zeros(len(self.Bx))
        self.Bmodel = np.zeros(len(self.t_pos))
        if len(self.t) == len(self.Bx):
            ' Length of t and Bx is equal. Proceed safely.'
        else:
            ' This is not safe. Len t and len Bx are NOT EQUAL. Please concider pulling out.'
        #Save untouched copy of data???
        if var.backupdata == True:
            self.t = self.t0
            self.Bx, self.By, self.Bz = self.Bx0, self.By0, self.Bz0
            #self.IGRFx, self.IGRFy, self.IGRFz = self.IGRFx0, self.IGRFy0, self.IGRFz0
        print ' Data loaded from files'
        return None
        
    def dump_to_file(self):
        '''Dump all data to file'''
        print ' Dumping all data to file.'
        data = self.t, self.Bx, self.By, self.Bz, self.t_pos, self.theta_usr, self.phi_usr, self.lat, self.lon, self.alt, self.B, self.Bmodel
        print ' Filename = project_dumped*.pyn'
        spec_name = str(raw_input(' * = '))
        np.save(var.dataname_dump + spec_name, data)
        return None
        
    def load_dumped_file(self):
        '''Loading all dumped data in a specific file'''
        spec_name = raw_input('''Filename of dumped file to load. 
             project_dumped*.pyn, * = ''')
        self.t, self.Bx, self.By, self.Bz, self.t_pos, self.theta_usr, self.phi_usr, self.lat, self.lon, self.alt, self.B, self.Bmodel = np.load(var.dataname_dump + spec_name +'.npy')
        return None
        
    def remove_png(self):
        print ''' Remove old png files? '''
        ans = raw_input('y/n ? ')
        if ans == 'y' or ans == 'Y':
            try:
                os.remove('graphs/*.png')
            except:
                print ''' Yheaaa, I need to debug this one day.. '''
        else:
            print ' ok, nothing removed.'
        return None
        



    def calc_B(self):
        ''' Calculating B = sqrt(Bx**2 + By**2 + Bz**2) '''
        print ''' Calculating B...'''
        self.B = np.zeros(np.size(self.Bx))
        BB = np.array((self.Bx, self.By, self.Bz))
        for i in range(len(self.B)):
            self.B[i] = np.linalg.norm(BB[:,i])
        return None
    
    def testB(self):
        



        return None

    def get_Bmodel(self):
        ''' Calculating theoretical B from models in IRBEMpy (spacepy) 
            based on lat, long and alt. Nothing more compleicated should 
            be neccecary '''
        print ' Calculating Bmodel '
        if self.range_set_done == True:
            print ''' Warning! You have changed the range. The Bmodel will not fully do so too. Show caution'''
        
        y = np.array([self.alt, self.lat, self.lon])
        y = np.ndarray.transpose(y) #getting the right dimensions for get_Bfield
        y = coord.Coords(np.array(y), 'GDZ', 'sph') 
        print 'y done'
        print self.alt

        launch = spacepy.time.Ticktock(var.launchtime)
        launchTAI = launch.TAI #convert to('TAI')
        launchTAII = launchTAI + self.t_pos
        T = spacepy.time.Ticktock(launchTAII, 'TAI')
        T = T.UTC
        t = spacepy.time.Ticktock(T, 'UTC' )
        print t
        #sys.exit()
        

        getB = spacepy.irbempy.get_Bfield(t,y, extMag='0')
        print ' get_Bfield ran. '

        self.Bmodel =  getB['Blocal']
        #print len(self.alt), len(self.lat)
        #print len(self.Bmodel), len(self.t), len(self.Bx), len(y), len(t), 'LENN'
        #mydata = spacedatamodel.SpaceData(attrs={'Blocal'}) #: 'BigSat1'})
        #print type(mydata), 'MYDATA' #data = dm.fromHDF5('test.h5')
        #self.Bmodel = getB.Blocal

        print ' Ran Bmodel. Dumping to file cause it takes TIME Give it a name.'
        spec_name = str(raw_input(' * = '))
        np.save(var.dataname_Bmodel_raw + spec_name, getB) #[getB['Blocal'], getB['Bvec']])

        return None

    
    def plot_magnetic(self):
        additional = raw_input(' plotting magn. additional name: ')
        Bs = [self.Bx, self.By, self.Bz, self.B] #, self.Bmodel]
        Bnames = ['Bx', 'By', 'Bz', 'B']#, 'Bmodel']
        for n in range(len(Bnames)):
            plt.plot(self.t, Bs[n])
            plt.savefig('graphs/plot%s%s.png' % (additional,Bnames[n]))
            plt.clf()
        plt.plot(self.t_pos, self.Bmodel)
        plt.savefig('graphs/plot%s%s.png' % (additional,'Bmodel'))
        plt.clf()
        return None
        
    def plot_position(self):
        additional = raw_input(' plotting pos. additional name: ')
        pos = [self.lat, self.lon, self.alt]
        posnames = ['Lat', 'Long', 'Alt']
        for n in range(len(posnames)):
            plt.plot(self.t_pos, pos[n])
            plt.savefig('graphs/plot%s%s.png' % (additional,posnames[n]))
            plt.clf()
        plt.plot(self.lat, self.lon)
        plt.savefig('graphs/plot%s%s.png' % (additional,'latlon'))
        plt.clf()
        return None


    def set_range(self):
        print ''' Setting range for (interesting)data '''
        self.range_set_done = True
        start = raw_input(' Begin at time ')
        stopp = raw_input(' End at time ')
        print type(self.t)
        print  self.t - 3 
        try:
            stab = np.argmin(abs(self.t - float(start)))
            stob = np.argmin(abs(self.t - float(stopp)))
            stap = np.argmin(abs(self.t_pos - float(start)))
            stop = np.argmin(abs(self.t_pos - float(stopp)))
        except:
            ''' Error. Went default. '''
            sta = np.argmin(abs(self.t - 70.))
            sto = np.argmin(abs(self.t - 250.))
            stap = np.argmin(abs(self.t_pos - 70.))
            stop = np.argmin(abs(self.t_pos - 250.))
        print len(self.t), len(self.t_pos)
        
        self.t = self.t[sta:sto]
        self.Bx = self.Bx[stab:stob]
        self.By = self.By[stab:stob]
        self.Bz = self.Bz[stab:stob]
        self.B = self.B[stab:stob]
        self.Bmodel = self.Bmodel[stab:stob]

        self.t_abs = self.t_abs[stap:stop]
        self.lon = self.lon[stap:stopp]
        self.lat = self.lat[stap:stop]
        self.alt = self.alt[stap:stop]
        print ''' New length of arrays: ''', len(self.t)
        return None


    def fft(self):
        
        #np.fft()
        return None
        
        
        
        
        
        
        
    def load_params(self):
        ''' To be removed? Not useful anymore '''
        # ca. t = 70 - 550 s
        self.start = np.argmin(abs(self.t-70.))
        self.stop = np.argmin(abs(self.t-550.))
        self.length = len(self.t)
        print self.length
        return None 
        
    








