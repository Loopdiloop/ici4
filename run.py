
# Running the code itself. Analysing the data.

'''     About the data:
    Boom fully deployed at 70.00000 s
    Data began ca. 10 sec before launch '''

# Standard software
import sys
import time

# My own imports
from data_processing import generate
import variables as var
from fetching import fetch

t_start = time.time()

fetishing = False
if fetishing == True:
    F = fetch()
    F.B()
    F.position()


run = generate()
#run.remove_png()
run.load()

run.get_Bmodel()

run.set_range()

run.median_filter()
run.despike_extreme()
run.inpaint()
#run.median_filter2()

run.plot_comparisons()

run.calc_B()

run.plot_magnetic()

#run.plot_magnetic() 
#run.plot_position()
run.dump_to_file()



#run.wavelet()
#run.despike()
#run.fill_data()

#run.plot_magnetic()

t_end = time.time()
t_tot = t_end - t_start
print '%.1f seconds' % t_tot



