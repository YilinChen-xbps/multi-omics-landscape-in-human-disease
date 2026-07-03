
import numpy as np
import pandas as pd
import statsmodels.api as sm
from tqdm import tqdm
from joblib import Parallel, delayed
import warnings
warnings.filterwarnings('error')

def results_summary(tgt_out_df):
    oratio_out_lst, p_out_lst = [], []
    for i in range(len(tgt_out_df)):
        oratio = f'{tgt_out_df.Odds_Ratio.iloc[i]:.2f}'
        lbd = f'{tgt_out_df.OR_LBD.iloc[i]:.2f}'
        ubd = f'{tgt_out_df.OR_UBD.iloc[i]:.2f}'
        oratio_out_lst.append(oratio + ' [' + lbd + '-' + ubd + ']')
    return oratio_out_lst

def process(omics_f, omics_df, tgt_df, my_cov_lst):
    tmp_omics_df = omics_df[['eid', omics_f]]
    tmp_df = pd.merge(tgt_df, tmp_omics_df, how='inner', on=['eid'])
    tmp_df.rename(columns={omics_f: 'x_omics'}, inplace=True)
    rm_eid_idx = tmp_df.index[tmp_df.x_omics.isnull() == True]
    tmp_df.drop(rm_eid_idx, axis=0, inplace=True)
    tmp_df.reset_index(inplace=True, drop=True)
    nb_all, nb_case = len(tmp_df), tmp_df.target_y.sum()
    prop_case = np.round(nb_case / nb_all * 100, 3)
    Y = tmp_df.target_y
    X = tmp_df[my_cov_lst + ['x_omics']]
    try:
        try:
            log_mod = sm.Logit(Y, sm.add_constant(X)).fit()
        except:
            log_mod = sm.Logit(Y, sm.add_constant(X)).fit(method='lbfgs')
        oratio = np.round(np.exp(log_mod.params).loc['x_omics'], 5)
        pval = log_mod.pvalues.loc['x_omics']
        ci_mod = log_mod.conf_int(alpha=0.05)
        lbd, ubd = np.round(np.exp(ci_mod.loc['x_omics'][0]), 5), np.round(np.exp(ci_mod.loc['x_omics'][1]), 5)
        tmpout = [omics_f, nb_all, nb_case, prop_case, oratio, lbd, ubd, pval]
    except:
        tmpout = [omics_f, nb_all, nb_case, prop_case, np.nan, np.nan, np.nan, np.nan]
    return tmpout

def read_tgt(dpath, tgt):
    tgt_df = pd.read_csv(dpath + 'Data/TargetData/TargetData/' + tgt + '.csv')
    rm_bl_idx = tgt_df.index[(tgt_df.target_y == 1) & (tgt_df.BL2Target_yrs > 0)]
    tgt_df.drop(rm_bl_idx, axis=0, inplace=True)
    tgt_df.reset_index(inplace=True, drop=True)
    return tgt_df

def get_cov_list(tgt_dict, tgt, cov_f_lst_all, cov_f_lst_non_sex):
    sex_id = tgt_dict.loc[tgt_dict.NAME == tgt].Sex.iloc[0]
    if (sex_id == 1) | (sex_id == 2):
        my_cov_lst = cov_f_lst_non_sex
    else:
        my_cov_lst = cov_f_lst_all
    return my_cov_lst

nb_cpus = 10

omics = 'Biochemistry'
group = 'Prevalent'
subgroup = 'All'


tgt_dict = pd.read_csv(dpath + '/Data/TargetData/DiseaseTable.csv', usecols = ['NAME', 'Sex', group+'Analysis'])
tgt_dict = tgt_dict.loc[tgt_dict[group+'Analysis'] == 1]
tgt_lst = tgt_dict.NAME.tolist()

omics_df = pd.read_csv(dpath + 'Data/BloodData/BiochemistryData.csv')
omics_f_lst = omics_df.columns[1:].tolist()

cov_df = pd.read_csv(dpath + '/Data/Covariates/Covariates.csv')
cov_df = cov_df.loc[(cov_df.Caucasian == 1)&(cov_df.Common != 1)]
cov_df['Age2'] = cov_df['Age']*cov_df['Age']
cov_df['AgeSex'] = cov_df['Age']*cov_df['Sex']
cov_df['Age2Sex'] = cov_df['Age2']*cov_df['Sex']
cov_df['Statin'].replace([np.nan, 0, 1], [-1, 0, 1], inplace = True)
cov_df[['Statin_NA', 'Statin_NO', 'Statin_YES']] = pd.get_dummies(cov_df['Statin'])
cov_f_lst = ['eid', 'Age', 'Sex', 'Age2', 'AgeSex', 'Age2Sex', 'TDI_imp', 'BMI_imp', 'Smoke_imp', 'FastingTime_imp', 'Season', 'Statin_NA', 'Statin_NO', 'Statin_YES']
cov_f_lst_all = ['Age', 'Sex', 'Age2', 'AgeSex', 'Age2Sex', 'TDI_imp', 'BMI_imp', 'Smoke_imp', 'FastingTime_imp', 'Season', 'Statin_NA', 'Statin_NO', 'Statin_YES']
cov_f_lst_non_sex = ['Age', 'Age2', 'TDI_imp', 'BMI_imp', 'Smoke_imp', 'FastingTime_imp', 'Season', 'Statin_NA', 'Statin_NO', 'Statin_YES']
cov_df = cov_df[cov_f_lst]


bad_tgt_lst = []

for tgt in tqdm(tgt_lst):
    try:
        my_cov_lst = get_cov_list(tgt_dict, tgt, cov_f_lst_all, cov_f_lst_non_sex)
        tgt_df = read_tgt(dpath, tgt)
        tgt_df = pd.merge(tgt_df, cov_df, how='inner', on=['eid'])
        tgt_out_df = Parallel(n_jobs=nb_cpus)(delayed(process)(omics_f, omics_df, tgt_df, my_cov_lst) for omics_f in omics_f_lst)
        tgt_out_df = pd.DataFrame(tgt_out_df)
        tgt_out_df.columns = ['Omics_f', 'NB_individual', 'NB_cases', 'Prop_Case(%)', 'Odds_Ratio', 'OR_LBD', 'OR_UBD', 'P_value']
        tgt_out_df['OR_output']= results_summary(tgt_out_df)
        tgt_out_df.to_csv(outpath + tgt + '.csv', index=False)
    except:
        bad_tgt_lst.append(tgt)

bad_df = pd.DataFrame(bad_tgt_lst)
bad_df.to_csv(badoutfile, index=False)

