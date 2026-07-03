
import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('error')

def results_summary(tgt_out_df):
    hr_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        hr = f'{tgt_out_df.Hazard_Ratio.iloc[i]:.2f}'
        lbd = f'{tgt_out_df.HR_LBD.iloc[i]:.2f}'
        ubd = f'{tgt_out_df.HR_UBD.iloc[i]:.2f}'
        hr_out_lst.append(hr + ' [' + lbd + '-' + ubd + ']')
    return hr_out_lst


def process(omics_f, omics_df, tgt_df, my_formula):
    tmp_omics_df = omics_df[['eid', omics_f]]
    tmp_df = pd.merge(tgt_df, tmp_omics_df, how='inner', on=['eid'])
    tmp_df.rename(columns={omics_f: 'x_omics'}, inplace=True)
    rm_eid_idx = tmp_df.index[tmp_df.x_omics.isnull() == True]
    tmp_df.drop(rm_eid_idx, axis=0, inplace=True)
    tmp_df.reset_index(inplace=True, drop=True)
    nb_all, nb_case = len(tmp_df), tmp_df.target_y.sum()
    prop_case = np.round(nb_case / nb_all * 100, 3)
    i, tmpout = 1, []
    while ((len(tmpout) == 0) & (i < 1e7)):
        cph = CoxPHFitter(penalizer=1e-7 * i)
        i = 10 * i
        try:
            cph.fit(tmp_df, duration_col='BL2Target_yrs', event_col='target_y', formula=my_formula)
            hr = np.round(cph.hazard_ratios_.x_omics, 5)
            lbd = np.round(np.exp(cph.confidence_intervals_).loc['x_omics'][0], 5)
            ubd = np.round(np.exp(cph.confidence_intervals_).loc['x_omics'][1], 5)
            pval = cph.summary.p.x_omics
            tmpout = [omics_f, nb_all, nb_case, prop_case, hr, lbd, ubd, pval]
        except:
            pass
    if tmpout == []:
        tmpout = [omics_f, nb_all, nb_case, prop_case, np.nan, np.nan, np.nan, np.nan]
    else:
        pass
    return tmpout

def read_tgt(dpath, tgt):
    tgt_df = pd.read_csv(dpath + 'Data/TargetData/TargetData/' + tgt + '.csv')
    rm_bl_idx = tgt_df.index[tgt_df.BL2Target_yrs <= 0]
    tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
    tgt_df.reset_index(inplace=True, drop=True)
    return tgt_df

def get_formula(tgt_dict, tgt, formula_full, formula_non_sex):
    sex_id = tgt_dict.loc[tgt_dict.NAME == tgt].Sex.iloc[0]
    if (sex_id == 1) | (sex_id == 2):
        my_formula = formula_non_sex
    else:
        my_formula = formula_full
    return my_formula

nb_cpus = 5

omics = 'Biochemistry'
group = 'Incident'
subgroup = 'All'



tgt_dict = pd.read_csv(dpath + '/Data/TargetData/DiseaseTable.csv', usecols = ['NAME', 'Sex', group+'Analysis'])
tgt_dict = tgt_dict.loc[tgt_dict[group+'Analysis'] == 1]
tgt_lst = tgt_dict.NAME.tolist()

omics_df = pd.read_csv(dpath + 'Data/BloodData/BiochemistryData.csv')
omics_f_lst = omics_df.columns[1:].tolist()

cov_df = pd.read_csv(dpath + '/Data/Covariates/Covariates.csv')
cov_df = cov_df.loc[(cov_df.Caucasian == 1)&(cov_df.Common != 1)]
cov_df.reset_index(drop = True, inplace = True)
cov_df['Statin'].replace([np.nan, 0, 1], [-1, 0, 1], inplace = True)
cov_f_lst = ['eid', 'Age', 'Sex', 'Race', 'TDI_imp', 'BMI_imp', 'Smoke_imp', 'FastingTime_imp', 'Season', 'Statin']
formula_full = "Age + C(Sex) + Age*Age + Age*C(Sex) + Age*Age*C(Sex) + TDI_imp + C(Smoke_imp) + BMI_imp + FastingTime_imp + C(Season) + C(Statin) + x_omics"
formula_non_sex = "Age + Age*Age + TDI_imp + C(Smoke_imp) + BMI_imp + FastingTime_imp + C(Season) + C(Statin) + x_omics"
cov_df = cov_df[cov_f_lst]


bad_tgt_lst = []

for tgt in tqdm(tgt_lst):
    try:
        my_formula = get_formula(tgt_dict, tgt, formula_full, formula_non_sex)
        tgt_df = read_tgt(dpath, tgt)
        tgt_df = pd.merge(tgt_df, cov_df, how='inner', on=['eid'])
        tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process)(omics_f, omics_df, tgt_df, my_formula) for omics_f in omics_f_lst)
        tgt_out_df = pd.DataFrame(tgt_out_df)
        tgt_out_df.columns = ['Omics_f', 'NB_individual', 'NB_cases', 'Prop_Case(%)', 'Hazard_Ratio', 'HR_LBD', 'HR_UBD', 'P_value']
        tgt_out_df['HR_output']= results_summary(tgt_out_df)
        tgt_out_df.to_csv(outpath + tgt + '.csv', index=False)
    except:
        bad_tgt_lst.append(tgt)

bad_df = pd.DataFrame(bad_tgt_lst)
bad_df.to_csv(badoutfile, index=False)

