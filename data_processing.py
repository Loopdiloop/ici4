
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
        self.Bmodel = np.zeros(len(self.Bx))
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

    def get_Bmodel(self):
        ''' Calculating theoretical B from models in IRBEMpy (spacepy) 
            based on lat, long and alt. Nothing more compleicated should 
            be neccecary '''
        print ' Calculating Bmodel '
        if self.range_set_done == True:
            print ''' Warning! You have changed the range. The Bmodel will not. Show caution'''
        t = spacepy.time.Ticktock(var.date0, 'ISO' )
        #self.t_abs, 'TAI') #Ticktock
        print 't done'
        #y = spacepy.coordinates.Coords([lon, lat, alt], 'SPH') 
        #GZD 
        y = coord.Coords([self.alt, self.lat, self.lon], dtype='GDZ', carsph='sph')
        #coord https://pythonhosted.org/SpacePy/quickstart.html
        print 'y done'
        Blocal, Bvec = spacepy.irbempy.get_Bfield(t,y)
        print ' Bvec etc ok! :D '
        print Bvec[:10]
        print np.shape(Bvec), 'BB FELTIIIK'
        sys.exit()
        #self.Bmodel = 
    
    
    
    def plot_magnetic(self):
        additional = raw_input(' plotting magn. additional name: ')
        Bs = [self.Bx, self.By, self.Bz, self.B, self.Bmodel]
        Bnames = ['Bx', 'By', 'Bz', 'B', 'Bmodel']
        for n in range(len(Bnames)):
            plt.plot(self.t, Bs[n])
            plt.savefig('graphs/plot%s%s.png' % (additional,Bnames[n]))
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
            sta = np.argmin(abs(self.t - float(start)))
            sto = np.argmin(abs(self.t - float(stopp)))
        except:
            ''' Error. Went default. '''
            sta = np.argmin(abs(self.t - 70.))
            sto = np.argmin(abs(self.t - 250.))
        print len(self.t)
        self.t = self.t[sta:sto]
        self.Bx = self.Bx[sta:sto]
        self.By = self.By[sta:sto]
        self.Bz = self.Bz[sta:sto]
        self.B = self.B[sta:sto]
        self.Bmodel = self.Bmodel[sta:sto]
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
        
    








