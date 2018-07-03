
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
#run.calc_B()
run.get_Bmodel()
run.plot_magnetic() #'project_data.txt')
run.plot_position()
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

