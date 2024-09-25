import numpy as np
import pandas as pd
import uproot

filename = '304014_user.nbruscin.26843628._000001.output.root'
file = uproot.open(filename)

tree = file['nominal_Loose']
feature_names = [
    'rapgap_higgsb_fwdjet',
    'bbs_top_m',
    'chi2_min',
    'njets_CBT2',
    'higgs_bb_m',
    'chi2_min_tophad_m',
    'chi2_min_tophad_pt',
    'chi2_min_tophad_eta',
    'nonbjets_eta'[0],
    'nonbjets_pt'[1],
    'chi2_min_Whad_pt',
    'rapgap_maxptjet',
    'chi2_min_bbnonbjet_m',
    'njets_CBT5',
    'chi2_min_higgs_m',
    # 'chi2_min_lmvmass_tH',
    'chi2_min_higgs_pt',
    'chi2_min_top_m',
    'chi2_min_top_pt',
    'chi2_min_DeltaPhi_tH',
    'chi2_min_DeltaEta_tH',
    'sphericity',
    #'alljet_m',
    'inv3jets',
    'rapgap_top_fwdjet',
    'nfwdjets',
    'nnonbjets',
    'chi2_min_deltaRq1q2',
    'foxWolfram_2_momentum',
    'foxWolfram_3_momentum'
]

branch = tree.arrays(feature_names, library="pd")

df = pd.DataFrame(branch)

# Save DataFrame to CSV file
df.to_csv('output.csv', index=False)


