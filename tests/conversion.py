import numpy as np
import pandas as pd
import uproot
import os

feature_names = [
    'rapgap_higgsb_fwdjet',
    'bbs_top_m',
    'chi2_min',
    'njets_CBT2',
    'higgs_bb_m',
    'chi2_min_tophad_m',
    'chi2_min_tophad_pt',
    'chi2_min_tophad_eta',
    'nonbjets_eta',
    'nonbjets_pt',
    'chi2_min_Whad_pt',
    'rapgap_maxptjet',
    'chi2_min_bbnonbjet_m',
    'njets_CBT5',
    'chi2_min_higgs_m',
    'chi2_min_lmvmass_tH',
    'chi2_min_higgs_pt',
    'chi2_min_top_m',
    'chi2_min_top_pt',
    'chi2_min_DeltaPhi_tH',
    'chi2_min_DeltaEta_tH',
    'sphericity',
    'alljet_m',
    'inv3jets',
    'rapgap_top_fwdjet',
    'nfwdjets',
    'nnonbjets',
    'chi2_min_deltaRq1q2',
    'foxWolfram_2_momentum',
    'foxWolfram_3_momentum'
]

dataset_id = 346676
dataset_name = 'tH'
directories_to_search = ['/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/tH_v34_minintuples_v0/mc16a_nom',
                       '/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/tH_v34_minintuples_v0/mc16d_nom',
                       '/eos/atlas/atlascerngroupdisk/phys-higgs/HSG8/tH_v34_minintuples_v0/mc16e_nom']

for directory in directories_to_search:
    for root, dirs, files in os.walk(directory):
        for file in files:
            if str(dataset_id) in file:
                opened = uproot.open(os.path.join(root, file))
                tree = opened['nominal_Loose']
                branches = tree.arrays(feature_names, library="pd")
                df = pd.DataFrame(branches)

                # Save DataFrame to CSV file
                df.to_csv(f'features/{dataset_name}.csv', index=False, mode='a')





