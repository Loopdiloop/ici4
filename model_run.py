# Standard software
import sys
import time

# My own imports
from data_processing import generate
import variables as var
from fetching import fetch

t_start = time.time()


run = generate()
run.load_dumped_file()
#run.fft_time()

run.set_range()
#run.wavelet()

run.fit_Bmodel()


#run.plot_comparisons()
run.plot_magnetic()
#run.plot_position()
#run.dump_to_file()


t_end = time.time()
t_tot = t_end - t_start
print '%.1f seconds' % t_tot