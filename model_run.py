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
#run.plot_magnetic()
run.set_range()
#run.fit_Bmodel()
#run.set_range()
#run.fft()
#run.wavelet()

run.init_fit_Bmodel()
run.fit_Bmodel()
'''
run.set_range()

'''

#run.plot_comparisons()

#run.plot_position()
#run.dump_to_file()


t_end = time.time()
t_tot = float(t_end - t_start)
print '%.1f sec per iteration' % (t_tot/var.fit_niter)
print '%.1f seconds total' % t_tot