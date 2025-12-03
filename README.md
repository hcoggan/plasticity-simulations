# plasticity-simulations
A repository of the code used to simulate bulk and single-cell sequencing of lung tumours under varying assumptions about plasticity and selection. 


### SIMULATION

* replacement_vpm_adjusted.py: BASE SIMULATION. Runs lung tumour evolution up to 5m cells with both genetic and nongenetic selection. Assumes nongenetic and genetic drivers at same strength and frequency.

* replacement_vpm_adjusted_vardeme.py: Accepts deme size as a variable parameter through argparse, allowing flexible control of the number of cells with which each cell is assumed to compete.

* replacement_vpm_adjusted_varrep.py: Accepts 'replacement rate' (maximum probability that one deme will become available to be replaced by another in any given timestep) as a variable parameter.

* replacement_vpm_adjusted_variable.py: Allows strength and frequency of nongenetic alterations to vary.

* replacement_vpm_with_survival_adjusted_single_cell.py: This produces single cell data from a single deme, up to 1000 cells.

### ANALYSIS

* july_single_cell_clone_clustering: the code used to analyse single-cell simulations.

* jul_compute_sumstats: the code used to extract summary statistics from simulations involving plasticity.

* jul_tracerx_analysis: the code used to extract summary statistics from TRACERx data.

* jan_classification_and_analysis: the code used to analyse bulk-tumour simulations.
