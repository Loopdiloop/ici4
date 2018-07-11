# My own imports
from data_processing import generate
import variables as var
from fetching import fetch

import time 

t_start = time.time()



run = generate()

run.load_dumped_file()
#run.plot_comparisons()
run.plot_magnetic()
run.plot_position()


t_end = time.time()
t_tot = t_end - t_start
print '%.1f seconds' % t_tot
