
# Launch in 2015

# Boom fully deployed at 70.00000 s
# Data began ca. 10 sek before launch


import numpy as np
import matplotlib.pyplot as plt
from Tkinter import *
import sys
import os
import pywt
import skimage.restoration as skres #for inpainting
import copy
import math

import spacepy
from spacepy import coordinates as coord
import spacepy.datamodel as spacedatamodel

import scipy
from scipy import signal
import scipy.signal as sign
import scipy.io as IO



#from initiate 
import variables as var
import minimizing as mini




class generate():
    def __init__(self):
        self.range_set_done = False
        if not os.path.exists('graphs'):
            os.makedirs('graphs')
        if not os.path.exists('dump_data'):
            os.makedirs('dump_data')
        if var.plot_comparison == True:
            self.plot_comp = {}
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
        self.tick = np.array(range(len(self.Bx)))
        if len(self.t) == len(self.Bx):
            ' Length of t and Bx is equal. Proceed safely.'
        else:
            ' This is not safe. Len t and len Bx are NOT EQUAL. Please concider pulling out.'
        #Save untouched copy of data???
        #if var.backupdata == True:
        #    self.t = self.t0
        #    self.Bx, self.By, self.Bz = self.Bx0, self.By0, self.Bz0
        if var.plot_comparison == True:
            #self.plot_comp_OG = np.zeros(len(self.By))
            #for i in range(len(self.By)):
            #    self.plot_comp_OG[i] = self.By[i]
            #self.plot_comp_OG = copy.deepcopy(self.By)
            self.plot_comp['OG'] = copy.deepcopy(self.By)
        print ' Data loaded from files' 
        return None 
        
    def dump_to_file(self):
        '''Dump all data to file'''
        print ' Dumping all data to file.'
        data = self.t, self.Bx, self.By, self.Bz, self.t_pos, self.theta_usr, self.phi_usr, self.lat, self.lon, self.alt, self.B, self.Bmodel, self.plot_comp
        print ' Filename = project_dumped*.pyn'
        spec_name = 'test_sofar_inpainted' #str(raw_input(' * = '))
        np.save(var.dataname_dump + spec_name, data)
        return None
        
    def load_dumped_file(self):
        '''Loading all dumped data in a specific file'''
        spec_name = 'test_sofar_inpainted' #raw_input('''Filename of dumped file to load. 
        #     project_dumped*.pyn, * = ''') # self.plot_comp
        self.t, self.Bx, self.By, self.Bz, self.t_pos, self.theta_usr, self.phi_usr, self.lat, self.lon, self.alt, self.B, self.Bmodel = np.load(var.dataname_dump + spec_name +'.npy')
        self.tick = np.array(range(len(self.Bx)))
        self.Bmodel_long = np.interp(self.t, self.t_pos, self.Bmodel)
        return None
        
    def remove_png(self):
        print ''' Remove old png files? '''
        ans = raw_input('y/n ? ')
        if ans == 'y' or ans == 'Y':
            try:
                os.remove('../graphs/*.png')
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
            self.B[i] = np.sqrt(self.Bx[i]**2 + self.By[i]**2 + self.Bz[i]**2)#np.linalg.norm(BB[:,i])
        self.Bmodel_long = np.interp(self.t, self.t_pos, self.Bmodel)
        return None
    
    def median_filter(self):
        self.medx = self.Bx - sign.medfilt(self.Bx, kernel_size=var.med_kernx)
        self.medy = self.By - sign.medfilt(self.By, kernel_size=var.med_kerny)
        self.medz = self.Bz - sign.medfilt(self.Bz, kernel_size=var.med_kernz)

        fig, axs = plt.subplots(3, 1, sharex=True)
        fig.subplots_adjust(hspace=0)
        axs[0].plot(self.t,np.log(self.medx))
        axs[1].plot(self.t, np.log(self.medy))
        axs[2].plot(self.t,np.log(self.medz))
        plt.title('Median filter minus the data')
        plt.xlabel('t[s]')
        plt.ylabel('log(signal - median)')
        plt.savefig('graphs/plot%s%s.png' % ('medX', 'medianfilter'))
        #plt.show()
        plt.clf()
        if var.plot_comparison == True:
            self.plot_comp['median_minus_By'] = copy.deepcopy(self.medy)
        return None

    def median_filter2(self):
        print ' median filter... '
        self.Bx = sign.medfilt(self.Bx, kernel_size=var.med_kernx)
        self.By = sign.medfilt(self.By, kernel_size=var.med_kerny)
        self.Bz = sign.medfilt(self.Bz, kernel_size=var.med_kernz)
        return None

    def despike_extreme(self):
        print ' despiking..'
        great = np.greater(self.medx, var.desp_x_up) + np.greater(self.medy, var.desp_y_up) + np.greater(self.medz, var.desp_z_up)
        less = np.greater(var.desp_x_down, self.medx) + np.greater(var.desp_y_down, self.medy) + np.greater(var.desp_z_down, self.medz)
        extreme = great + less
        for i in range(len(great)):
            if extreme[i] == True:
                self.Bx[i-1:i+1] = float('nan')
                self.By[i-1:i+1] = float('nan')
                self.Bz[i-1:i+1] = float('nan')
        if var.plot_comparison == True:
            self.plot_comp['despiked'] = copy.deepcopy(self.By)
        return None

    def inpaint(self):
        print ' Beginning inpainting. '
        mask = np.zeros(len(self.By))
        for i in range(len(self.By)):
            if math.isnan(self.Bx[i]) or math.isnan(self.By[i]) or math.isnan(self.Bz[i]):
                mask[i] = 1

        self.Bx = skres.inpaint_biharmonic(self.Bx, mask, multichannel=False) ; print ' Bx done'        
        self.By = skres.inpaint_biharmonic(self.By, mask, multichannel=False) ; print ' By done'    
        self.Bz = skres.inpaint_biharmonic(self.Bz, mask, multichannel=False) ; print ' Bz done'    
        if var.plot_comparison == True:
            self.plot_comp['inpainted'] = copy.deepcopy(self.By)
        print ' Inpainting done!'
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

        launch_realtimeTAI = var.launch_TAI + self.t_pos
        T = spacepy.time.Ticktock(launch_realtimeTAI, 'TAI')
        T = T.UTC
        t = spacepy.time.Ticktock(T, 'UTC' )
        
        getB = spacepy.irbempy.get_Bfield(t,y, extMag='0')
        print ' get_Bfield ran. '

        self.Bmodel =  getB['Blocal']
        print ' Ran Bmodel. Dumping to file cause it takes TIME Give it a name.'
        spec_name = 'testing' #str(raw_input(' * = '))
        np.save(var.dataname_Bmodel_raw + spec_name, getB)
        return None



    def fit_Bmodel(self):
        # 3rd order : ax**2 + bx + c

        test_waytoogood = np.array([ 2.05422143e-05, -4.15690381e+04, 1.03866410e-05, 
            -1.54944104e+04, 1.08294995e-05, -4.59599374e+04])

        Bx = self.Bx ; By = self.By ; Bz = self.Bz
        Bmodel_long = self.Bmodel_long
        minimize_1st_order = lambda x: abs(np.sum(np.sqrt((x[0]*Bx + x[1])**2 + (x[2]*By + x[3])**2 
            + (x[4]*Bz + x[5])**2) - Bmodel_long))
        
        minimize_2nd_order = lambda x: abs(np.sum(np.sqrt((x[0]*Bx**2 + x[1]*Bx + x[2])**2 + 
            (x[3]*By**2 + x[4]*By + x[5])**2 + (x[6]*Bz**2 + x[7]*Bz + x[8])**2) - Bmodel_long))

        minimize_3rd_order = lambda x: abs(np.sum(np.sqrt((x[0]*Bx**3 + x[1]*Bx**2 + x[2]*Bx + x[3])**2 + 
            (x[4]*By**3 + x[5]*By**2 + x[6]*By + x[7])**2 + 
            (x[8]*Bz**3 + x[9]*Bz**2 + x[10]*Bz + x[11])**2) - Bmodel_long))

        initial_guess_1st_order = np.array([mini.XC1, 0., mini.YC1, 0., 
            mini.ZC1, 0. ]).astype(float)

        initial_guess_1st_order_David = np.array([mini.XC1, mini.XD1, mini.YC1, mini.YD1, 
            mini.ZC1, mini.ZD1 ]).astype(float)
            
        '''np.array([  5.40800596e-06,  -7.46808905e+03,   9.74210627e-04,
        -6.92984651e+05,   1.29220637e-05,  -3.18654229e+04]) '''
        
        '''np.array([ 2.05422143e-05, -4.15690381e+04, 1.03866410e-05, 
            -1.54944104e+04, 1.08294995e-05, -4.59599374e+04])'''
        '''np.array([mini.XC1, mini.XD1, mini.YC1, mini.YD1, 
            mini.ZC1, mini.ZD1 ]).astype(float)'''

        initial_guess_2nd_order = np.array([mini.XB1, mini.XC1, mini.XD1, mini.YB1, 
            mini.YC1, mini.YD1, mini.ZB1, mini.ZC1, mini.ZD1 ]).astype(float)

        initial_guess_3rd_order = np.array([mini.XA1, mini.XB1, mini.XC1, mini.XD1, 
            mini.YA1, mini.YB1, mini.YC1, mini.YD1, mini.ZA1, mini.ZB1, mini.ZC1, 
            mini.ZD1 ]).astype(float)



        minimize_1st_order_full = lambda x: np.sqrt((x[0]*Bx + x[1])**2 + (x[2]*By + x[3])**2 
            + (x[4]*Bz + x[5])**2)# - Bmodel_long
        

        self.XX = minimize_1st_order_full(initial_guess_1st_order_David)
        fs = 5000./ (self.t[5000] - self.t[0])
        N = len(self.B)
        B_dict = dict({'B': self.B, 'firstorder':self.XX, 'fs': fs, 'N': N})
        IO.savemat('B.mat', B_dict)
        IO.savemat('B.mat', B_dict)
        print ' saved, ok'
        '''
        ress = minimize_1st_order_full(test_waytoogood)
        plt.plot(self.t, self.B) 
        plt.plot(np.linspace(self.t[0], self.t[-1], len(ress)), ress)  
        plt.plot(self.t, self.Bmodel_long)  
        plt.show()
        plt.clf()'''
        #sys.exit()
        
        print ' Running fitting of Bmodel, %s iterations' % var.fit_niter
        #result = scipy.optimize.basinhopping(minimize_1st_order, initial_guess_1st_order, 
        #    var.fit_niter, var.fit_T , var.fit_stepsize)

        '''Outout result: 
        The optimization result represented as a OptimizeResult object. 
        Important attributes are: x the solution array, fun the value of 
        the function at the solution, and message which describes the cause 
        of the termination. The OptimzeResult object returned by the selected 
        minimizer at the lowest minimum is also contained within this 
        object and can be accessed through the lowest_optimization_result attribute.'''   
        
        #print result
        #np.save(var.fit_datadump, result)

        return None


    def set_range(self):
        print ''' Setting range for (interesting)data '''
        self.range_set_done = True
        start = 400 #raw_input(' Begin at time ')
        stopp = 460 #raw_input(' End at time ')
        try:
            stab = np.argmin(abs(self.t - float(start)))
            stob = np.argmin(abs(self.t - float(stopp)))
            stap = np.argmin(abs(self.t_pos - float(start)))
            stop = np.argmin(abs(self.t_pos - float(stopp)))
        except:
            ''' Error. Went default. '''
            stab = np.argmin(abs(self.t - 70.))
            stob = np.argmin(abs(self.t - 250.))
            stap = np.argmin(abs(self.t_pos - 70.))
            stop = np.argmin(abs(self.t_pos - 250.))
        #print stap, stopp
        
        self.t = self.t[stab:stob]
        self.Bx, self.By, self.Bz = self.Bx[stab:stob], self.By[stab:stob], self.Bz[stab:stob]
        self.B, self.Bmodel = self.B[stab:stob], self.Bmodel[stab:stob]
        self.tick = self.tick[stab:stob]
        self.Bmodel_long = self.Bmodel_long[stab:stob]

        try:
            self.t_abs = self.t_abs[stap:stop]
        except:
            print 'no t_abs, but it ok, little you'
        self.lon = self.lon[stap:stop] ; self.lat = self.lat[stap:stop] ; self.alt = self.alt[stap:stop]
        if var.plot_comparison == True:
            #By_og = copy.deepcopy(self.By_OG)
            self.plot_comp_OG = self.plot_comp_OG[stab:stob]  #plot_comp['OG'][stab:stob]
        print ''' New length of arrays: ''', len(self.t)
        return None


    def plot_magnetic(self):
        additional = 'raw data' #raw_input(' plotting magn. additional name: ')
        Bs = [self.Bx, self.By, self.Bz, self.B] #, self.Bmodel]
        Bnames = ['Bx', 'By', 'Bz', 'B']#, 'Bmodel']
        for n in range(len(Bnames)):
            plt.plot(self.t, Bs[n])#, 'b-') #, '*')
            plt.title('ici-4 raw data for reference of magnetic field By.', fontsize=30)#'Plot ')#%s %s' % (additional,Bnames[n]))
            #'ici-4 raw data of magnetic field By.'
            plt.text(0.95, 0.05, 'Preliminary results', fontsize=40, color='red', ha='right', va='bottom', alpha=0.3)
            plt.ylabel('Signal, nT ', fontsize=20)
            plt.xlabel('time [s]') #Ticks (approx. after launch)', fontsize=20)
            plt.savefig('graphs/plot_raw_data_%s_zoom' % Bnames[n])#%s%s.png' % (additional,Bnames[n]))
            plt.show()
            plt.clf()

        print len(self.t_pos), len(self.Bmodel)
        #plt.plot(self.t_pos, self.Bmodel)
        #plt.savefig('graphs/plot%s%s.png' % (additional,'Bmodel'))
        #plt.clf()

        print len(self.t_pos), len(self.Bmodel)
        #plt.plot(self.t, self.B, 'r')
        #plt.plot(self.t_pos, self.Bmodel,'b') #is behind.. 
        #plt.savefig('graphs/plot%s%s.png' % (additional,'BmodelB'))
        #plt.clf()
        return 
        
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

    def plot_comparisons(self):
        
        fig, axs = plt.subplots(4, 1, sharex=True)
        fig.subplots_adjust(hspace=0)

        fig.suptitle('Comparisons of raw data and stages of corrections.', fontsize=24)
        #fig.grid(color='r', linestyle='-', linewidth=2)

        axs[0].plot(self.tick, self.plot_comp_OG, label='Original data')
        axs[0].set_xlim(var.plot_comp_xlim)
        axs[0].set_ylim([7.3e8, 8.0e8])
        axs[0].legend()
        axs[0].grid('on')

        axs[1].plot(self.tick, self.plot_comp['median_minus_By'], label='Median - By')
        axs[1].set_xlim(var.plot_comp_xlim)
        axs[1].set_ylim([-1e9, 3.5e9])
        axs[1].legend()
        
        axs[2].plot(self.tick, self.plot_comp['despiked'], label='Despiked data')
        axs[2].set_xlim(var.plot_comp_xlim)
        axs[2].set_ylim(var.plot_comp_ylim)
        axs[2].legend()

        axs[3].plot(self.tick, self.plot_comp['inpainted'], label='Inpainted')
        axs[3].set_xlim(var.plot_comp_xlim)
        axs[3].set_ylim(var.plot_comp_ylim)
        axs[3].legend()

        fig.text(0.95, 0.05, 'Preliminary results', fontsize=40, color='red', ha='right', va='bottom', alpha=0.3)
        axs[1].set_ylabel('Signal, nT (?)', fontsize=20)
        plt.xlabel('Ticks (approx. after launch)', fontsize=20)
        
        plt.savefig('graphs/plot%s.png' % ('method_comparison'))
        #plt.show()
        plt.clf()

        return None


    def fft(self):
        '''
        for m in [self.Bx, self.By, self.Bz, self.B]:
            fft = abs(np.fft.rfft(m))
            plt.plot(np.linspace(0,1,len(fft)), fft)
            plt.show()
            plt.clf
        '''
        X = self.B
        fmax = 200
        fmin = 0
        fs = 5000./(self.t[5000] - self.t[0]) #freq
        width = 2000  /2
        jump = 1500
        map = [] #np.zeros((bins, fft_len))
        step = width
        #stop = start + width
        while step+(width) < len(X):
            map.append(np.real(np.fft.rfft(X[step-width:step+width])))
            
            step += jump
        map = np.array(map).astype(float)
        #print map
        plt.pcolormesh(map, vmin=-1., vmax=1., cmap='ocean')
        plt.show()




        return None

    def fft_time(self):
        fs = 5000./ (self.t[5000] - self.t[0])
        print fs
        nper = 300
        B_dict = dict({'B': self.B})
        IO.savemat('B.mat', B_dict)
        IO.savemat('B.mat', B_dict)
        '''
        for fs in [20, 30, 50, 70, 100, 120, 150, 300, 500, 701, 2000, 3000, 4000]:
            #f,t,Zxx = signal.stft(self.B, fs, nperseg=nper)
            f, t, Zxx = signal.spectrogram(self.By, fs)
            
            plt.pcolormesh(t, f, np.abs(Zxx))#, vmin=0, vmax=4000)
            plt.title('STFT Magnitude, %s ' % nper)
            plt.ylabel('Frequency [Hz]')
            plt.xlabel('Time [sec]')
            plt.savefig('graphs/bbbb%s.png' % nper)
            plt.clf()'''
        
        print 'ok, everything ran'
        
        '''
        y = self.By
        powerSpectrum, freqenciesFound, time, imageAxis = plt.specgram(y, Fs=fs, cmap='rainbow')
        #Pxx, freqs, bins, im = plt.specgram(signal, Fs=fs)
        plt.ylim((0,70))
        plt.xlabel('Time')
        plt.ylabel('Frequency')
        plt.show() '''


        '''
        freqs, times, Sx = signal.spectrogram(audio, fs=rate, window='hanning', 
            nperseg=1024, noverlap=M - 100, detrend=False, scaling='spectrum')

        f, ax = plt.subplots(figsize=(4.8, 2.4))
        ax.pcolormesh(times, freqs / 1000, 10 * np.log10(Sx), cmap='viridis')
        ax.set_ylabel('Frequency [kHz]')
        ax.set_xlabel('Time [s]');'''

        return None
        
    def wavelet(self):

        '''t = np.linspace(-1, 1, 200, endpoint=False)
        sig  = np.cos(2 * np.pi * 7 * t) + signal.gausspulse(t - 0.4, fc=2)
        widths = np.arange(1, 31)
        cwtmatr = signal.cwt(sig, signal.ricker, widths)
        plt.imshow(cwtmatr, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.show()'''
        
        '''widths = np.arange(1, 21)
        cwtmatr = signal.cwt(self.B, signal.ricker, widths)
        plt.imshow(cwtmatr, extent=[self.t[0], self.t[-1], 1, 21], cmap='PRGn', aspect='auto',
            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
        plt.show()'''

        #wav = pywt.Wavelet('mor1')
        #phi, psi, x = wav.wavefun(level=5)
        
        '''
        import numpy as np

        import matplotlib.pyplot as plt

        import pywt

        # use 'list' to get a list of all available 1d demo signals
        signals = pywt.data.demo_signal('list')

        subplots_per_fig = 5
        signal_length = len(self.B)
        i_fig = 0
        n_figures = int(np.ceil(len(signals)/subplots_per_fig))
        for i_fig in range(n_figures):
            # Select a subset of functions for the current plot
            func_subset = signals[
                i_fig * subplots_per_fig:(i_fig + 1) * subplots_per_fig]

            # create a figure to hold this subset of the functions
            fig, axes = plt.subplots(subplots_per_fig, 1)
            axes = axes.ravel()
            for n, signal in enumerate(func_subset):
                if signal in ['Gabor', 'sineoneoverx']:
                    # user cannot specify a length for these two
                    x = pywt.data.demo_signal(signal)
                else:
                    x = pywt.data.demo_signal(signal, signal_length)
                ax = axes[n]
                ax.plot(x.real)
                if signal == 'Gabor':
                    # The Gabor signal is complex-valued
                    ax.plot(x.imag)
                    ax.legend(['Gabor (Re)', 'Gabor (Im)'], loc='upper left')
                else:
                    ax.legend([signal, ], loc='upper left')
            # omit axes for any unused subplots
            for n in range(n + 1, len(axes)):
                axes[n].set_axis_off()
        plt.show()





        #import the libraries
        import matplotlib.pyplot as plot
        import numpy as np

        # Define the list of frequencies
        frequencies         = np.arange(5,105,5)

        # Sampling Frequency
        samplingFrequency   = 400

        # Create two ndarrays
        s1 = np.empty([0]) # For samples
        s2 = np.empty([0]) # For signal
        # Start Value of the sample
        start   = 1
        # Stop Value of the sample
        stop    = samplingFrequency+1

        for frequency in frequencies:
            sub1 = np.arange(start, stop, 1)
            # Signal - Sine wave with varying frequency + Noise
            sub2 = np.sin(2*np.pi*sub1*frequency*1/samplingFrequency)+np.random.randn(len(sub1))
            s1      = np.append(s1, sub1)
            s2      = np.append(s2, sub2)

            start   = stop+1
            stop    = start+samplingFrequency

        # Plot the signal
        plot.subplot(211)
        plot.plot(s1,s2)
        plot.xlabel('Sample')
        plot.ylabel('Amplitude')

        # Plot the spectrogram
        plot.subplot(212)
        powerSpectrum, freqenciesFound, time, imageAxis = plot.specgram(s2, Fs=samplingFrequency)
        plot.xlabel('Time')
        plot.ylabel('Frequency')
        plot.show()   '''

        return None


    def save_matlab(self):
        return None
        


