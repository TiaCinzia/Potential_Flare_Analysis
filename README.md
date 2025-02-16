# Potential_Flare_Analysis

This page contains all the data, code and information generated/required for the presentation of my dissertation: "Solar Flare Energetic Particles: Where are electrons in-situ accelerated".

'Raw_Data' is the data downloaded directly from INSPEX into a .txt file, for each presented flare, numbered 1-9. It was not possible to use this raw data directly in Juptyer Notebook since some data was listed as dates or instrument names (since data was downloaded from STEP on SolO). Flare 7 was omitted from the Flare Table present in the planning document as the data could not clearly be downloaded using INSPEX.

'Processed_Data' contains the .txt files that were used alongside the python code to generate the flare spectra of flux versus energy. Data processing included removing 'unrecognised' elements from the raw data file - such as names and dashes (-) which would otherwise make the file unreadable.

'Python_Files' includes each .ipynb Jupyter Notebook file that was used to generate the final plots that are present on this poster. There is one file for each flare, which outputs the flux-energies plot. They match to flares 1-9 in the suggested flare table, with the omission of flare 7 for reasons previously stated.

Feel free to have a closer look at the data and code used to make this portion of the project possible! :)
