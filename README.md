# plasticity-simulations
A repository of the code used to simulate bulk and single-cell sequencing of lung tumours under varying assumptions about plasticity and selection. 


### SIMULATION

* baseline_bulk_tumour_simulation.py: BASE SIMULATION. Runs lung tumour evolution up to 5m cells with both genetic and nongenetic selection. Assumes nongenetic and genetic drivers at same strength and frequency.

* simulate_with_variable_deme_size.py: Accepts deme size as a variable parameter through argparse, allowing flexible control of the number of cells with which each cell is assumed to compete.

* simulate_with_variable_replacement_rate.py: Accepts 'replacement rate' (maximum probability that one deme will become available to be replaced by another in any given timestep) as a variable parameter.

* simulate_with_variable_semiheritable_strength.py: Allows strength and frequency of nongenetic alterations to vary.

* simulate_with_single_cell_sampling.py: Simulates tumour evolution and produces single cell data from a single deme, up to 1000 cells.

### ANALYSIS

* analyse_single_cell_simulations.ipynb: the code used to analyse single-cell simulations.

* get_summary_statistics_from_simulated_tumours.ipynb: the code used to extract summary statistics from simulations involving plasticity.

* get_summary_statistics_from_real_tumours.ipynb: the code used to extract summary statistics from TRACERx data.

* analyse_simulations_and_produce_figures.ipynb: the code used to analyse bulk-tumour simulations and produce all main and supplementary figures related to the modelling. To reproduce these figures, download all files in main-tumour-inputs.zip and other-necessary-inputs.zip and add them to a single directory. You can then run the Jupyter notebook with this directory as your base filepath.
