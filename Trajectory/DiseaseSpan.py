
import glob
import re
import os
import scipy.stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from skmisc.loess import loess
pd.options.mode.chained_assignment = None  # default='warn'

def sort_nicely(l):
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c.replace("_","")) for c in re.split('([0-9]+)', key) ]
    l.sort( key=alphanum_key )
    return l


def process(mydf, timelines_pred, my_alpha, my_f):
    tmpdf = mydf[['Year', my_f]].copy()
    tmpdf.dropna(inplace=True)
    tmpdf.reset_index(inplace=True, drop=True)
    loess_m = loess(tmpdf['Year'], tmpdf[my_f], span=my_alpha, surface='direct')
    loess_m.fit()
    pred_f = loess_m.predict(timelines_pred, stderror=True)
    pred_val = pred_f.values
    pred_lbd = pred_f.confidence().lower
    pred_ubd = pred_f.confidence().upper
    tmpout_df = pd.DataFrame({my_f + '_loess_fit': pred_val,
                              my_f + '_loess_lbd': pred_lbd,
                              my_f + '_loess_ubd': pred_ubd})
    return tmpout_df


my_alpha = 1


timelines_pred = [-15 + 0.5*i for i in range(31)]

for zscore_dir in tqdm(zscore_dir_lst[:400]):
    tmp_basename = os.path.basename(zscore_dir)
    tmp_basename = tmp_basename.replace('trajectories_Zscore_', 'Loess_fitted_')
    tmp_outfile_name = tmp_basename.replace('tsv', 'csv')
    tmp_outfile_dir = outpath + tmp_outfile_name
    myout_df = pd.DataFrame({'timeline': timelines_pred})
    tmp_df = pd.read_csv(zscore_dir, sep = '\t')
    tmp_df = tmp_df.loc[tmp_df.Year<0]
    tmp_df.reset_index(inplace = True, drop = True)
    tmp_f_lst = tmp_df.columns.tolist()[3:]
    for tmp_f in tqdm(tmp_f_lst):
        tmpout_df = process(tmp_df, timelines_pred, my_alpha, tmp_f)
        myout_df = pd.concat([myout_df, tmpout_df], axis=1)
    myout_df.to_csv(tmp_outfile_dir, index = False)



my_alpha = 1
datype = 'Biochemistry'
my_f_lst = mydf.columns.tolist()[2:]
timelines_pred = [40+0.5*i for i in range(61)]
myout_df = pd.DataFrame({'timeline': timelines_pred})

for my_f in tqdm(my_f_lst):
    tmpout_df = process(mydf, timelines_pred, my_alpha, my_f)
    myout_df = pd.concat([myout_df, tmpout_df], axis=1)

myout_df.to_csv(outpath + '/' + datype + '_alpha'+str(my_alpha)+'.csv', index = False)

