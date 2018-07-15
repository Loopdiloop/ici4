# Junk code





test_waytoogood = np.array([ 2.05422143e-05, -4.15690381e+04, 1.03866410e-05, 
            -1.54944104e+04, 1.08294995e-05, -4.59599374e+04])








            
#deltafit = 3 #delta allowed for finished fit
#fit_order = 5 #order of fittings. 5 = 0th to 4th order.
fit_param_c = [300000, 40000, 4000]
fit_param_b = [3, 4, 4]
fit_param_a = [3, 4, 4]
'''fit_param_bc = [fit_param_b[0], fit_param_c[0], 
    fit_param_b[1], fit_param_c[1],
    fit_param_b[2], fit_param_c[2]]'''
fit_param_bc = [fit_param_b[0], fit_param_b[1],
    fit_param_b[2], fit_param_c[0]]

'''fit_param_abc = [fit_param_a[0], fit_param_b[0], fit_param_c[0], 
    fit_param_a[1], fit_param_b[1], fit_param_c[1],
    fit_param_a[2], fit_param_b[2], fit_param_c[2]]'''
fit_param_abc = [fit_param_a[0], fit_param_b[0], 
    fit_param_a[1], fit_param_b[1],
    fit_param_a[2], fit_param_b[2], fit_param_c[0]]



# param_bc
        #minimize_2nd_order = lambda A: np.sum(np.sqrt( A[0]*self.Bx + A[1]*self.By + A[2]*self.Bz + A[3]) - self.Bmodel_long)
        # param_abs
        #minimize_3rd_order = lambda x: np.sum(np.sqrt(x[0]*self.Bx**2 + x[1]*self.Bx
        #    + x[2]*self.By**2 + x[3]*self.By + x[4]*self.Bz**2 + x[5]*self.Bz + x[6]) - self.Bmodel_long)

        #minimize_abc_nosum = lambda x: np.sqrt(x[0]*self.Bx**2 + x[1]*self.Bx
        #    + x[2]*self.By**2 + x[3]*self.By + x[4]*self.Bz**2 + x[5]*self.Bz + x[6]) - self.Bmodel_long

        #plt.plot(self.t, self.Bmodel_long)
        #for i in np.linspace(10, 50, 15):
        #   #print minimize_c(i)
        #    plt.plot(self.t, minimize_abc_nosum([-7.97645071e+02, 3.0, -7.94306010e+01, 4.0, -6.45352895e+02, 4.0*i**2, 3.0e5])) #, legend='%s'%i)
        #plt.legend()

        #plt.show()
        #lt.clf
        #sys.exit()
        # '''
        #last_result = np.load(var.fit_datadump)
        
        # Last result: np.array([ -7.97645071e+02, 3.0, -7.94306010e+01, 4.0, -6.45352895e+02, 4.0, 3.0e5])
        #initial_guess = last_result['x'] #var.fit_param_abc
        #initial_guess = np.array([ -7.97645071e+02, 3.0, -7.94306010e+01, 4.0, -6.45352895e+02, 4.0, 3.0e5])
        
        
        


        #(self.By, fs=1.0, window='hann', nperseg=256, noverlap=None, nfft=None, detrend=False, return_onesided=True, boundary='zeros', padded=True, axis=-1))

        #for m in [self.Bx, self.By, self.Bz, self.B]:
            #fft = (np.fft.rfft2(m))
            #plt.pcolormesh(fft)
            #plt.show()
            #plt.clf
        #plt.pcolormesh()




    def fill_data(self):
        #to find avg total in three 1D params
        delta_avg = np.array([np.sum(abs(self.Bx - np.roll(self.Bx, 1)))/len(self.Bx),
            np.sum(abs(self.By - np.roll(self.By, 1)))/len(self.By),
            np.sum(abs(self.Bz - np.roll(self.Bz, 1)))/len(self.Bz) ])
        #delta_avg /= len(self.Bx)
        print delta_avg
        for m in [self.Bx, self.By, self.Bz]:
            for j in range(var.fit_order):
                deltas += var.fit_param[j]*(np.sum(abs(m - np.roll(m, j))) + np.sum(abs(m - np.roll(m, -j))))/(2*np.len(m))
                
                #if deltas >= deltas_tolerance:
                    
        avg_delta = np.sum([3,4])/len(self.Bx)
        return None




        #cA, cD = pywt.dwt(self.Bx, 'db2')
        #x2 = pywt.idwt(cA, cD, 'db2')
        #CA, CD = np.meshgrid(cA, cD)
        #plt.pcolor(CA, CD, np.linspace(0,100,101))
        #plt.show()
        
        '''
        import pylab 
        import scipy.io.wavfile as wavfile
        x = self.Bx
        # Find the highest power of two less than or equal to the input.
        def lepow2(x):
            return 2 ** floor(log2(x))

        # Make a scalogram given an MRA tree.
        def scalogram(data):
            bottom = 0

            vmin = min(map(lambda x: min(abs(x)), data))
            vmax = max(map(lambda x: max(abs(x)), data))

            gca().set_autoscale_on(False)

            for row in range(0, len(data)):
                scale = 2.0 ** (row - len(data))

                imshow(
                    array([abs(data[row])]),
                    interpolation = 'nearest',
                    vmin = vmin,
                    vmax = vmax,
                    extent = [0, 1, bottom, bottom + scale])

                bottom += scale

        # Load the signal, take the first channel, limit length to a power of 2 for simplicity.
        rate, signal = wavfile.read('kitten.wav')
        signal = signal[0:lepow2(len(signal)),0]
        tree = pywt.wavedec(signal, 'db5')

        # Plotting.
        pylab.gray()
        pylab.scalogram(tree)
        pylab.show() '''

        '''
        from scipy import fftpack, ndimage
        import matplotlib.pyplot as plt

        image = ndimage.imread('image2.jpg', flatten=True)     # flatten=True gives a greyscale image
        fft2 = fftpack.fft2(image)

        plt.imshow(fft2)
        plt.show() '''
        
        
        
        
    def load_params(self):
        ''' To be removed? Not useful anymore '''
        # ca. t = 70 - 550 s
        self.start = np.argmin(abs(self.t-70.))
        self.stop = np.argmin(abs(self.t-550.))
        self.length = len(self.t)
        print self.length
        return None 
        

        A = np.zeros((9))
        A0 = var.fit_param_abc
        A_c = np.zeros((3))
        A_c0 = var.fit_param_c
        


 #np.array([2,3,4,5,6,7,2,4,1000])
        #Bx_dir = A[0]*self.Bx**2 + A[1]*self.Bx + A[2]
        #Bx_dir = A[3]*self.By**2 + A[4]*self.By + A[5]
        #Bx_dir = A[6]*self.Bz**2 + A[7]*self.Bz + A[8]
        #fun = lambda ABC: ABC[0]*x**2  ABC[1]*x + ABC[2] - B_model_long
        
        #haha nope. wanna see something nasty? Keep readin the next few lines of code!
        '''parameters = []
        j = 0
        for a0 in [5., 10, 100]:
            for a1 in [5., 100]:
                for a2 in [1000, 20000]:
                    for a3 in [5., 10, 100]:
                        for a4 in [5., 100]:
                            for a5 in [1000, 20000]:
                                for a6 in [5., 10, 100]:
                                    for a7 in [5., 100]:
                                        for a8 in [1000, 20000]: 
                                            result = 0 
                                            for i in range(len(self.Bx)):
                                                result += np.sqrt(a0*self.Bx[i]**2 + a1*self.Bx[i] + a2 + a3*self.By[i]**2 + a4*self.By[i] + a5 + a6*self.Bz[i]**2 + a7*self.Bz[i] + a8) - Bmodel_long[i]
                                            parameters.append([float(result), a0, a1, a2, a3, a4, a5, a6, a7, a8])
                            print j
                            j +=1 
        print parameters
        np.save('parameters, bruteforce', parameters)
        '''


#res_bc = 
        #res_abc = 
        

        '''
        #3rd order
        #A, B, C =
        x = self.B 
        print float(len(x)) / len(self.Bmodel), float(len(x)) / len(self.Bmodel) - 28*len(self.Bmodel), len(self.alt)
        B_model_long = np.repeat(self.Bmodel,28, axis=0)
        print self.Bmodel[:100]
        print 'ok'
        #B_model_long = np.
        
        fun = lambda ABC: ABC[0]*x**2  ABC[1]*x + ABC[2] - B_model_long
        #A*x**2 + B*x + C = B_model
        F = scipy.optimize.minimize(fun, [1e3, 1e3, 3.5e9], args=(), method='CG', )
        print F
        exit()'''








###COPY




            #if self.Bx[i] == float('nan') or self.By[i] == float('nan') or self.Bz[i] == float('nan'):




        #print 'y done'
        #print self.alt

        #launch = spacepy.time.Ticktock(var.launch_print)
        #launchTAI = launch.TAI #convert to('TAI')
        launch_realtimeTAI = var.launch_TAI + self.t_pos
        T = spacepy.time.Ticktock(launch_realtimeTAI, 'TAI')
        T = T.UTC
        t = spacepy.time.Ticktock(T, 'UTC' )
        

        getB = spacepy.irbempy.get_Bfield(t,y, extMag='0')
        print ' get_Bfield ran. '

        self.Bmodel =  getB['Blocal']
        #print len(self.alt), len(self.lat)
        #print len(self.Bmodel), len(self.t), len(self.Bx), len(y), len(t), 'LENN'
        #mydata = spacedatamodel.SpaceData(attrs={'Blocal'}) #: 'BigSat1'})
        #print type(mydata), 'MYDATA' #data = dm.fromHDF5('test.h5')
        #self.Bmodel = getB.Blocal








        sys.exit()
        #axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
        #axs[0].set_ylim(-1, 1)
        #axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
        #axs[1].set_ylim(0, 1)
        #axs[2].set_yticks(np.arange(-0.9, 1.0, 0.4))
        #axs[2].set_ylim(-1, 1)
        
        plt.savefig('graphs/plot%s%s.png' % ('medX', 'medianfilter'))
        plt.clf()
        sys.exit() 



        plt.plot(self.t, self.medx)
        plt.plot(self.t, self.medy)
        plt.plot(self.t, self.medz)
        plt.savefig('graphs/plot%s%s.png' % ('medX', 'medianfilter'))
        plt.show()
        plt.clf()
        sys.exit()


        #T = np.zeros(len(Bx))
        #t = np.array(tick3)

        #for j in range(len(t)):
        #    T[t_num[j]] = t[j]
        #minn = int(0.08*len(Bx))
        #T = np.linspace(T[0], np.max(T[-minn-100:-minn]), len(Bx)-minn)

        '''t = np.array(tick2)
        plt.plot(t, np.linspace(0,len(t),len(t)))
        plt.show()
        sys.exit()
        '''


        #plt.plot(By, tick3)
        #plt.show()
        #sys.exit()
        #t_abs = tick3
        #t_abs = np.linspace(t_abs[0], t_abs[-1], len(Bx)) #absolute length if wanted..

        #tick3 = np.array(tick3) - float(tick[0]) + 1.

        #T = np.linspace(tick3[0], np.max(tick3[-100: ]), len(Bx), dtype=float)
        plt.plot(T, np.linspace(0, len(T), len(T)))
        plt.show()
        sys.exit()





        '''tt = spacepy.time.Ticktock(t)
        t = tt.TAI
        t -= t[0] 
        T = np.zeros(len(Bx))
        for j in range(len(t)):
            T[t_num[j]] = t[j]
        minn = int(0.08*len(Bx))
        T = np.linspace(T[0], np.max(T[-minn-100:-minn]), len(Bx)-minn)
        '''



####### TICK TOCK MF

        #t = spacepy.time.Ticktock(var.date0, 'ISO' )
        #use TAI?

        #t_mod = [] ; y = []
        #for l in range(len(self.t_abs)):
        #    if l // 300 == 0 :
        #        t_mod.append(self.t_abs[l])
        #        y.append(np.array([self.lon[l], self.lat[l], self.alt[l]]))
        #t_mod = np.array(t_modd)

        #TEST THIS SHIT
        #ttt = spacepy.time.Ticktock(['2002-02-02T12:00:00', '2002-02-02T12:10:00'], 'ISO')
        #yyy = coord.Coords([[3,0,0],[2,0,0]], 'GEO', 'car')
        #fff = spacepy.irbempy.get_Bfield(ttt,yyy)
        #print fff 
        #sys.exit()

#([self.alt, self.lat, self.lon], dtype='GDZ', carsph='sph')
        #coord https://pythonhosted.org/SpacePy/quickstart.html

    def fetch_olddata(self): #, filename):
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
        

    def fetch_B(self):
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
                t.append(ny[0]) ; t_num.append(f)
                Bx.append(ny[1]) ; By.append(ny[2]) ; Bz.append(ny[3])
            else:
                asc_len = False
            f +=1
            
        # NEXT LINES OF CODE ARE UGLY AF PLS IGNORE MUST FIX LATER 
        # LINEAR FIT FOR TIME, IS ALMOST PERFECT THO
        #translate T!!!1 + fill out (linearly)
        tt = spacepy.time.Ticktock(t)
        t = tt.TAI
        t -= t[0] 
        T = np.zeros(len(Bx))
        for j in range(len(t)):
            T[t_num[j]] = t[j]
        minn = int(0.08*len(Bx))
        T = np.linspace(T[0], np.max(T[-minn-100:-minn]), len(Bx)-minn)
        
        print ' Read from file ', var.filename_B
        
        proj_data = np.array(T), np.array(Bx), np.array(By), np.array(Bz)
        np.save(var.dataname_B, proj_data)
        print 'Saved in file ', var.dataname_B
        return None 
        

    def fetch_position_vec(self):
        '''Opens filename (20151022ICI4.dat) and saves it in an npy file.
        should be made more general.'''
        files = open(var.filename_pos, 'r')
        asc_len = True
        
        t = [] ; t_num = []
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
            print ny
            sys.exit()

            
            f +=1
            
        # NEXT LINES OF CODE ARE UGLY AF PLS IGNORE MUST FIX LATER 
        # LINEAR FIT FOR TIME, IS ALMOST PERFECT THO
        #translate T!!!1 + fill out (linearly)
        '''tt = spacepy.time.Ticktock(t)
        t = tt.TAI
        t -= t[0] 
        T = np.zeros(len(Bx))
        for j in range(len(t)):
            T[t_num[j]] = t[j]
        minn = int(0.08*len(Bx))
        T = np.linspace(T[0], np.max(T[-minn-100:-minn]), len(Bx)-minn)
        '''
        print ' Read from file ', var.filename_pos
        
        #proj_data = np.array(T), np.array(Bx), np.array(By), np.array(Bz)
        #np.save(var.dataname_B, proj_data)
        print 'Saved in file ', var.dataname_pos
        return None 




https://guides.github.com/activities/hello-world/

'''
plt.plot(t, Bx)
plt.show()
plt.clf()


kk = np.fft.fft(Bx)
plt.plot(t, kk)
plt.show()
plt.clf()
    
print len(Bx), len(IGRFx)
print 'lensSSSSS'

#proj_data = np.linspace(0,1,3)

#np.save('project_data', proj_data)


print dictlab, 'dict'


# 



Time = np.zeros(L)
Port0 = np.zeros(L)
stream = np.zeros(L)
Frame = np.zeros(L)
Tempint = np.zeros(L)
Tempext = np.zeros(L)
Magn = np.zeros(L)
AccY = np.zeros(L)
AccX = np.zeros(L)
Photo = np.zeros(L)
P = np.zeros(L)
Batt = np.zeros(L)


def plot(array0, array1, title, ylab, xlab, name):

    plt.plot(array0, array1)
    plt.title(title)
    plt.xlabel(xlab)
    plt.ylabel(ylab)
    plt.grid()
    #plt.show()
    plt.savefig(name)
    plt.clf()

var= file.readline()
var = var.replace('[-]','')
var = var.replace('__Raw','')
var = map(str, var.split())
print var



k=0
while k < L: #length == True:
    array = file.readline()
    data_raw = (map(float, array.split()))
    try:
        #Raw = (data_raw)
        Time[k], stream[k], Frame[k], Tempint[k], Tempext[k], Magn[k], AccX[k], AccY[k], Photo[k], P[k], Batt[k] = data_raw
    except:
        if len(data_raw) == 2:
            Time[k], stream[k] = data_raw        
            print 'tiny error'        
        else:
            Time[k] = Time[k-1]
    if F == 0:
        if Time[k] > 158.5:
            F=k
        
    k+=1
    print k
if F == 0:
    print 'ERROR'
print Time[-5]

#plot(Time, Magn, 'magn.png', K)
#plot(Time, Tempint, 'tempint.png', K)
#plot(Time, Tempext, 'tempext.png', K)
#plot(Time, Photo, 'photo.png', K)
#plot(Time, P, 'pressure.png', K)


Port0 = np.zeros(L)
stream = np.zeros(L)
Frame = np.zeros(L)
Tempint = Tempint*0.178
Tempext = Tempext*0.4
Magn = Magn*0.02
AccY = AccY*0.49 - 62.745
AccX = AccX*0.980 - 125.490
Photo = Photo*0.02
P = P*4.357 + 111.024
Batt = Batt*0.084


plot(Time[F:L], Magn[F:L], 'Magnetometer', 'Time', '[Volts]', 'magn.png')
plot(Time[F:L], Tempint[F:L], 'Internal temperature', '[Time]', '[C]', 'tempint.png')
plot(Time[F:L], Tempext[F:L], 'External temperature', '[Time]', '[C]','tempext.png')
plot(Time[F:L], Photo[F:L], 'Photo sensor', '[Time]', '[Volts]', 'photo.png')
plot(Time[F:L], P[F:L], 'Pressure', '[Time]', '[mBar]','pressure.png')
plot(Time[F:L], AccX[F:L], 'Acceleration, X', '[Time]', '[m/s]', 'accx.png')
plot(Time[F:L], AccY[F:L], 'Acceleration, Y', '[Time]', '[m/s]', 'accy.png')
#plot(Time[F:L], Batt[F:L], 'Battery power', '[Time]', '[V]', 'batt.png')






#554144



#outfile = ('outfile.txt', 'w')
#outfile.write()
'''


#k=1000000
        #print 'Error'

    #if len(Raw)<5:
    #        K = k
    #        k = 1000000
    #        break
    #if 8:
    #    for m in range(10):
    #        print 'K:', k
    #        array = file.readline()
    #        data_raw = (map(float, array.split()))
    #        print data_raw
    #        k+=1
        
    #Time[k], stream[k], Frame[k], Tempint[k], Tempext[k], Magn[k], AccX[k], AccY, Photo[k], P[k], Batt[k] = Raw





# http://turl.no/1g78 

