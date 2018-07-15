# ici4
My summer project of 2018 working on data from magnetometer from the sounding rocket ici-4
(REPOSITORY NO. 2 SINCE I HAd A BIT OF A FIGHT WITH GIT :( :( )

IS THE CODE SOMEHOW USEFUL?
- Yes, for people working on this data specifically and probably no one else.

SO MARIANNE, HOW BUGGED IS IT ATM?
- Very, I would not reccomend running it and hoping for something useful. It is also highly fragmented and sectional, so you do need a slight bit of insight into how it's built to run it succsessfully. There is also big experimental chunks and a wavelet + a time dependent "sonogram"-style plot that spews out nonsense.

WHAT THE CODE DOES:
- For now it's not only a framework for loading the data and basic plotting, but now it can despike raw data, run a median filter, inpaint what will end up missing. On top of this it can also make a B-field model for the rockets trajectory and to a 1st, 2nd AND 3rd order fit of the data. (the crowd goes mild). It looks shamefully ugly right now, but relax, it's under highly experimental constuction with basically no supervision (please save me, I'm running around blindly).

HOW TO RUN:
- Beware, frustration ahead (nah, it should run mostly ok.(UPDATE: not really))
- Run the file run.py (>> python run.py) (or any other file with "run" in the name)
- run.py (any run file) decides what to run of the other files with actual functions. It's basically the coordinator of the rest.
- Run calls upon data_processing.py and/or fetching.py (or any of the other new file additions)
- fetching.py can fetch data from the raw data files and makes .pyn-files with all the data.
- data_processing.py contains all the actual plotting, calculating and magnetic models. Will be heavily expanded.

It may still hard for anyone other than the author to run. Will hopefully change in the future as documentation will be made and debugging will happen. Cheers.
