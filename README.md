# Potential_Flare_Analysis

This page contains all the data, code and information generated/required for the presentation of my dissertation: **"Solar Flare Energetic Particles: Where are electrons in-situ accelerated"**.

**'inspex_v55.py'** is the current version of INSPEX that has been used in this project, developed by Samuel Carter. Since the code is still in development, version 55 has been used so far, and the most up-to-date version will be used for the processing of the chosen flare. The code takes advantage of the Spyder user interface and is able to generate time series and spectra for flares at a chosen interval and specified peak. 

**'Raw_Data'** is the data downloaded directly from INSPEX into a .txt file, for each presented flare, numbered 1-9. It was not possible to use this raw data directly in Juptyer Notebook since some data was listed as dates or instrument names (since data was downloaded from STEP on SolO). Flare 7 was omitted from the Flare Table present in the planning document as the data could not clearly be downloaded using INSPEX.

**'Processed_Data'** contains the .txt files that were used alongside the python code to generate the flare spectra of flux versus energy. Data processing included removing 'unrecognised' elements from the raw data file - such as names and dashes (-) which would otherwise make the file unreadable.

**'Python_Files'** includes each .ipynb Jupyter Notebook file that was used to generate the final plots that are present on this poster. There is one file for each flare, which outputs the flux-energies plot. They match to flares 1-9 in the suggested flare table, with the omission of flare 7 for reasons previously stated.

Feel free to have a closer look at the data and code used to make this portion of the project possible! :)
