
''' Variables etc for data_processing0.py and run.py.
    Avoiding hardcoding this too bad. '''

import spacepy    


#Save backupdata? 
backupdata = False

#Plot sammenlikning av for/etter div. median etc.
plot_comparison = True

# Original .asc file containing data. 
filename_old = '../data/ICI4_magn_hires.asc'
filename_B = '../data/190215_launch_fgm_xyz.asc'
filename_pos = '../data/20151022ICI4.dat'

# No. of lines of header before one line content of file.
header_length_old = 5
header_length_B = 0
header_length_pos = 11

# *.npy file with the generated data from datafile.
dataname_old = '../data/project_data_old'
dataname_B = '../data/project_data_B'
dataname_pos = '../data/project_data_pos'

dataname_dump = '../data/project_dumped'
dataname_Bmodel_raw ='../data/project_Bmodel_raw'


#For the Ticktock Bmodel
#Copypaste from datafile: 

t_B0_ISO = '2015-02-19T22:03:46.982095' #'22:03:46.982095'
t_B0_ = spacepy.time.Ticktock(t_B0_ISO, 'ISO')
t_B0_TAI = t_B0_.TAI

t_B1_ISO = '2015-02-19T22:07:05.051936' #'22:07:05.051936' #32
t_B1_ = spacepy.time.Ticktock(t_B1_ISO, 'ISO')
t_B1_TAI = t_B1_.TAI

#launchtime = '19.02.2015T22:06:41' #UTC
launch_ISO = '2015-02-19T22:06:41' #ISO
launch_ticktock = spacepy.time.Ticktock(launch_ISO, 'ISO')
launch_TAI = launch_ticktock.TAI

#print 'TAI'
#print launch_TAI
#print t_B0_TAI, t_B1_TAI


# For fitting of data-params:
deltafit = 3 #delta allowed for finished fit
fit_order = 5 #order of fittings. 5 = 0th to 4th order.
fit_param = [1., 0.6, 0.3, 0.1, 0., 0., 0., ]



# Median filter:
med_kernx = 41
med_kerny = 41
med_kernz = 41

# For the despiking. Cut evertythgin that is 
# spiking more than this in median data
desp_x_up = 2e7 #2000 #8e7
desp_y_up = 2e7 #2000 #7e7
desp_z_up = 2e7 #2000 #7e7
desp_x_down = -2e7 #8e7
desp_y_down = -2e7 #7e7
desp_z_down = -2e7 #7e7

#plot comparisons
plot_comp_xlim = [680.50, 682.50]
plot_comp_ylim = [7.3e8, 8.0e8]



