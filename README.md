# ici4
My summer project of 2018 working on data from magnetometer from the sounding rocket ici-4
REPOSITORY NO. 2 SINCE I HAVE A BIT OF A FIGHT WITH GIT :( :(


SO MARIANNE, HOW BUGGED IS IT ATM?
- Very, I would not reccomend running it and hoping for something useful.

WHAT THE CODE DOES:
- For now it's only still a framework for loading the data and basic plotting of raw data.

HOW TO RUN:
- Beware, frustration ahead (nah, it should run mostly ok.)
- Run the file run.py (>> python run.py)
- run.py decides what to run of the other files with actual functions. It's basically the coordinator of the rest.
- Run calls upon data_processing.py and/or fetching.py
- fetching.py can fetch data from the raw data files and makes .pyn-files with all the data.
- data_processing.py contains all the actual plotting, calculating and magnetic models. Will be heavily expanded.

It may still hard for anyone other than the author to run. Will hopefully change in the future as documentation will be made and debugging will happen.
