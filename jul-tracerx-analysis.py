import numpy as np
from matplotlib import pyplot as plt
from numba import njit
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib as mpl
import pandas as pd

base_filepath = "/Users/44749/Documents/PhD_work/datasets_for_analysis"

#Extract summary statistics and clinical variables from real TRACERx tumours.

#read in mutations and correct for clonality
df = pd.read_csv("C:/Users/44749/Documents/PhD_work/datasets_for_analysis/full_tx421primary_muttable_vaf_ccf_clonality.csv")

print("Table loaded")

filtered_df= df[df['PASS'] == True & (df['Timing_SC'] != "E")]

unique_tumours = filtered_df[['cruk_tumour_id']].drop_duplicates()



#translate table into dictionary

driver_mut_dict = {True:1, False:0}
all_tumour_names = []

for tumour in unique_tumours.values.tolist():
    tID = tumour[0]
    all_drivs_found = []
    clonal_appearances_of_drivers = {}
    all_tumour_names.append(tID)
    df_here = filtered_df[filtered_df['cruk_tumour_id'] == tID]
    unique_regions = df_here[['region']].drop_duplicates()
    #print(unique_regions)
    tumour_regions = [{} for m in range(len(unique_regions.values.tolist()))]
    #print(unique_regions)
    for n, region in enumerate(unique_regions.values.tolist()):
        regID = region[0]
        #print(tID, regID)
        relevant_muts = df_here[df_here['region']==regID]
        #print(len(relevant_muts))
        for index, row in relevant_muts.iterrows():
            mutID = str(row['chr']) + "." +str(row['start']) + "." + str(row['ref']) + "." + str(row['var']) + ":" + str(row['Hugo_Symbol']) + ":" + str(driver_mut_dict[row['DriverMut']])
            ccf = min(1, float(row['final_ccf'])) if row['cluster_clonality'] != 'clonal' else 1.0
            #print(tID, regID, mutID, ccf)
            #print(regID, ccf)
            if ccf > 0:
                tumour_regions[n][mutID] = ccf
                if row['DriverMut']:
                    all_drivs_found.append(row['Hugo_Symbol']) #add driver to list if it's found here, might be added more than once- record only the Hugo Symbol, NOT the specific mutation
                    #print(mutID)
                    if ccf == 1.0:
                        if row['Hugo_Symbol'] in clonal_appearances_of_drivers:
                            clonal_appearances_of_drivers[row['Hugo_Symbol']] += 1
                        else:
                            clonal_appearances_of_drivers[row['Hugo_Symbol']] = 1 #keep track of whether drivers are clonal
        # print(tumour_regions[n])
    np.save(tID+"_mutdict.npy", tumour_regions)

    #now check whether drivers are clonal or subclonal
    num_regions = len(unique_regions)
    all_drivs_found = np.unique(all_drivs_found)
    clonal_drivers = []
    subclonal_drivers = []
    
    for driv in all_drivs_found:
        if driv in clonal_appearances_of_drivers:
            print(driv, clonal_appearances_of_drivers[driv], num_regions)
            if clonal_appearances_of_drivers[driv] == num_regions: #found clonally everywhere
                clonal_drivers.append(driv)
        else:
            subclonal_drivers.append(driv)
    
    np.save(base_filepath+"/"+tID+"_driver_muts.npy", all_drivs_found)
    print(tID, "clonal drivers", clonal_drivers)
    np.save(base_filepath+"/"+tID+"_clonal_driver_muts.npy", clonal_drivers)
    np.save(base_filepath+"/"+tID+"_subclonal_driver_muts.npy", subclonal_drivers)

np.save("all_tumour_names.npy", all_tumour_names)
    

    df = pd.read_excel("C:/Users/44749/Documents/PhD_work/datasets_for_analysis/tracerx_clinical_df.xlsx")
purity_df = pd.read_excel("C:/Users/44749/Documents/PhD_work/datasets_for_analysis/tx421_purity_ploidy.xlsx")

sex_codes = {'Male':0, 'Female':1}
histology_codes = {'LUAD':0, 'LUSC':1, 'Other':2}
smok_stat_codes = {'Smoker':0, 'Ex-Smoker':1, 'Never Smoked':2}

num_samples = np.load("num_samples_by_tumour_name.npy", allow_pickle=True).item()


patient_index_dict = {} #record where all patients are in this list
pcount = 0

clinical_dfs = []

unique_drivs = []
for name in list(num_samples.keys()):
    #load drivers
    drivers_here = np.load(base_filepath+"/"+name+"_driver_muts.npy", allow_pickle=True)
    #print(drivers_here)
    unique_drivs += list(drivers_here)

all_drivs = np.unique(unique_drivs)
#assign each an index
driv_index_dict = dict(zip(list(all_drivs), list(range(len(all_drivs)))))
np.save("all_driv_identities.npy", all_drivs)

num_drivs = len(all_drivs)
driver_matrix = []
    

for name in list(num_samples.keys()):
    try:
        drivers_here = np.load(base_filepath+"/"+name+"_driver_muts.npy", allow_pickle=True)
        clonal_drivers_here = np.load(base_filepath+"/"+name+"_clonal_driver_muts.npy", allow_pickle=True)
        subclonal_drivers_here = np.load(base_filepath+"/"+name+"_subclonal_driver_muts.npy", allow_pickle=True)
        #print(name, "clonal driver check", clonal_drivers_here, len(clonal_drivers_here))
        
        age = df[df['tumour_id_muttable_cruk']==name]['age']
        #print(name, age)
        #print(age.iloc[0]
        age = age.iloc[0]
    
        sex = df[df['tumour_id_muttable_cruk']==name]['clinical_sex']
        sex = str(sex.iloc[0])
        bin_sex = sex_codes[sex]
        #print(bin_sex)
        clinical_sex = bin_sex
    
        hist = str(df[df['tumour_id_muttable_cruk']==name]['histology_3'].iloc[0])
        int_hist = histology_codes[hist]
        histology = int_hist
    
        smok_stat = str(df[df['tumour_id_muttable_cruk']==name]['smoking_status_merged'].iloc[0])
        smoking_status = smok_stat_codes[smok_stat]

        pyears = df[df['tumour_id_muttable_cruk']==name]['pack_years']
        pack_years= int(pyears.iloc[0])
    
        patient_index_dict[name] = pcount
    
        #calculate average/minimum purity of all regions in the tumour
        av_pur = np.average(purity_df[purity_df['tumour_id']==name]['Purity'])
        min_pur = np.min(purity_df[purity_df['tumour_id']==name]['Purity'])
        
        clinical_dfs.append([age, clinical_sex, histology, smoking_status, pack_years, len(clonal_drivers_here), len(subclonal_drivers_here), av_pur, min_pur])
        
        driver_matrix_here = np.zeros(num_drivs, dtype=int)
        for driv in list(drivers_here):
            index = driv_index_dict[driv]
            driver_matrix_here[index] = 1 #mark the presence of all detected drivers
    
        driver_matrix.append(driver_matrix_here)

        

        pcount += 1
        

    except:
        print(name)

np.save("patient_index_dict.npy", patient_index_dict)
np.save("clinical_var_matrix.npy", clinical_dfs)
np.save("clinical_driver_matrix.npy", driver_matrix)

