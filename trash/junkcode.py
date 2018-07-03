# Junk code


###COPY


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

