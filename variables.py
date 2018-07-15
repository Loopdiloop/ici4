
''' Variables etc for data_processing0.py and run.py.
    Avoiding hardcoding this too bad. '''

import spacepy    
import numpy as np


#Save backupdata? 
backupdata = False

#Plot sammenlikning av for/etter div. median etc.
plot_comparison = False



# For fitting of data-params:

fit_niter = 2
fit_T = 50 
fit_stepsize = 1000
fit_datadump = '../dump_data/Bmodel_fit'


# Median filter:
med_kernx = 111 #41
med_kerny = 111
med_kernz = 111

# For the despiking. Cut evertythgin that is 
# spiking more than this in median data
desp_x_up = 2e7 #2000 #8e7
desp_y_up = 2e7 #2000 #7e7
desp_z_up = 2e7 #2000 #7e7
desp_x_down = -2e7 #8e7
desp_y_down = -2e7 #7e7
desp_z_down = -2e7 #7e7

#plot comparisons
plot_comp_xlim = [1389000, 1396000] #[486.50, 488.25] #[680.50, 682.50]
plot_comp_ylim = [7.4e8, 8.0e8]



# FILENAMES ETC

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

dataname_dump = '../dump_data/project_dumped'
dataname_Bmodel_raw ='../dump_data/project_Bmodel_raw'


#For the Ticktock Bmodel

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


