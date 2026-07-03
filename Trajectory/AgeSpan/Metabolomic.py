
import scipy.stats
import numpy as np
import pandas as pd
from tqdm import tqdm
from skmisc.loess import loess

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
datype = 'Metabolomic'
my_f_lst = mydf.columns.tolist()[2:]
timelines_pred = [40+0.5*i for i in range(61)]
myout_df = pd.DataFrame({'timeline': timelines_pred})

i = 0
for my_f in tqdm(my_f_lst):
    tmpout_df = process(mydf, timelines_pred, my_alpha, my_f)
    myout_df = pd.concat([myout_df, tmpout_df], axis=1)
    if i%10 == 0:
        myout_df.to_csv(outpath + '/' + datype + '_alpha' + str(my_alpha) + '.csv', index=False)
    else:
        pass
    i+=1


myout_df.to_csv(outpath + '/' + datype + '_alpha'+str(my_alpha)+'.csv', index = False)

#!/bin/bash
#SBATCH -p DCU         # Queue
#SBATCH -N 1          # Node count required for the job
#SBATCH -n 1          # Number of tasks to be launched
#SBATCH --mem=32G
#SBATCH -J age_pro          # Job name
#SBATCH -o age_pro.out       # Standard output
#SBATCH -w node7

cd /home1/jiayou/Documents/Projects/MultiOmics/Results/Trajectory/Code/
python AgeProteomic.py