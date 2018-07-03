
''' Variables etc for data_processing0.py and run.py.
    Avoiding hardcoding this too bad. '''
    


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





#For the Ticktock Bmodel
#Copypaste from datafile: 
date0 = '22:03:46.982095'
date1 = '22:07:05.051936'




