
''' Variables etc for data_processing0.py and run.py.
    Avoiding hardcoding this too bad. '''

import spacepy    


#Save backupdata? 
backupdata = False

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

print 'TAI'
print launch_TAI
print t_B0_TAI, t_B1_TAI


