
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

fetishing = True


if fetishing == True:
    F = fetch()
    F.B()
    F.position()


run = generate()
#run.remove_png()
run.load()
#run.set_range()
run.despike_extreme()
run.median_filter()
run.calc_B()
run.get_Bmodel()
run.plot_magnetic() #'project_data.txt')
#run.plot_position()
#run.wavelet()
#run.despike()
#run.fill_data()
#run.set_range()
#run.plot_magnetic()


'''
ran = generate()
print 'cross fingers'
ran.load() '''

#print run.content

t_end = time.time()
t_tot = t_end - t_start
print '%.1f seconds' % t_tot


# For fitting of data-params:
deltafit = 3 #delta allowed for finished fit
fit_order = 5 #order of fittings. 5 = 0th to 4th order.
fit_param = [1., 0.6, 0.3, 0.1, 0., 0., 0., ]