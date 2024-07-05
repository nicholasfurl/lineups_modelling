#!/usr/bin/env python
# coding: utf-8

# In[361]:


import pandas as pd


# In[362]:


lin = pd.read_excel('data_lineups_python_ROCtry.xlsx')


# In[363]:


# hits
lin.loc[(lin['Decision'] == lin['correct_answer']) & (lin['target_presence'] == 'tp'), 'hits'] = 1


# In[364]:


# average hits over tp trials for participants
tp_data = lin[lin['target_presence'] == 'tp']
hits_grps = tp_data.groupby(['distinctiveness', 'delay', 'Participant Private ID'])['hits'].mean().reset_index()
mean_hits = hits_grps['hits']


# In[365]:


# false alarms 
lin.loc[(lin['Decision'] != 'Reject') & (lin['correct_answer'] == 'Reject'), 'fas'] = 1


# In[366]:


# average fas over ta trials for participants
ta_data = lin[lin['target_presence'] == 'ta']
fas_grps = ta_data.groupby(['distinctiveness', 'delay', 'Participant Private ID'])['fas'].mean().reset_index()
mean_fas = fas_grps['fas']


# In[367]:


mean_fas_div = mean_fas/5


# In[368]:


# deal with extreme values for dprime and c - MacMillan & Kaplan, 1985
N = 21

hits_adj = mean_hits.values.copy()
fas_adj = mean_fas_div.values.copy()

hits_adj[mean_hits == 0] = 1 / (2 * N)
hits_adj[mean_hits == 1] = 1 - 1 / (2 * N)

fas_adj[mean_fas_div == 0] = 1 / (2 * N)
fas_adj[mean_fas_div == 1] = 1 - 1 / (2 * N)


# In[369]:


print(hits_adj)


# In[370]:


print(fas_adj)


# In[371]:


import scipy as sc
from scipy.stats import norm


# In[372]:


dprime_ss = norm.ppf(hits_adj) - norm.ppf(fas_adj)
print(dprime_ss)


# In[373]:


c_ss = (norm.ppf(hits_adj) + norm.ppf(fas_adj)) / 0.5
print(c_ss)


# In[374]:


import os


# In[375]:


current_dir = os.getcwd()
output_csv_filename = "lin_mean_hits.csv"
output_csv_path = os.path.join(current_dir, output_csv_filename)
hits_grps.to_csv(output_csv_path, index=True)


# In[376]:


current_dir = os.getcwd()
output_csv_filename = "lin_mean_fas_div.csv"
output_csv_path = os.path.join(current_dir, output_csv_filename)
mean_fas_div.to_csv(output_csv_path, index=True)


# In[377]:


from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt


# In[378]:


# misses 
lin.loc[(lin['Decision'] == 'Reject') & (lin['target_presence'] == 'tp'), 'misses'] = 1
tp_data = lin[lin['target_presence'] == 'tp']
misses_grps = tp_data.groupby(['distinctiveness', 'delay', 'Participant Private ID'])['misses'].mean().reset_index()
mean_misses = misses_grps['misses']
print(misses_grps)


# In[379]:


# correct rejections
lin.loc[(lin['Decision'] == 'Reject') & (lin['correct_answer'] == 'Reject'), 'corr_rej'] = 1
ta_data = lin[lin['target_presence'] == 'ta']
correct_rej_grps = ta_data.groupby(['distinctiveness', 'delay', 'Participant Private ID'])['corr_rej'].mean().reset_index()
correct_rej_grps['corr_rej'] = correct_rej_grps['corr_rej'].fillna(0)
mean_correct_rej = correct_rej_grps['corr_rej']
print(correct_rej_grps)


# In[380]:


# define levels of confidence, distinctiveness, and delay
confidence_levels = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
distinctiveness_levels = ["veridical", "caricature", "anticaricature"]
delay_levels = ["delay", "no delay"]


# In[381]:


# iterate over distinctiveness levels
for dist_level in distinctiveness_levels:
    # iterate over delay levels
    for del_level in delay_levels:
        # subset data for the current distinctiveness and delay levels
        subset = lin[(lin['distinctiveness'] == dist_level) & (lin['delay'] == delay_level)]


# In[382]:


# initialize lists to store fpr, tpr, and auc for each confidence level
fpr_list = []
tpr_list = []
auc_list = []


# In[383]:


# iterate over confidence levels
for conf_level in confidence_levels:
    # Subset data for the current confidence level
    subset_conf = subset[subset['Confidence'] >= conf_level]


# In[388]:


lin['mean_hits_list'] = hits_grps['hits']
lin ['mean_fas_div_list'] = fas_grps['fas']/5
lin ['mean_misses_list'] = misses_grps['misses']
lin ['mean_correct_rej_list'] = correct_rej_grps['corr_rej']


# In[395]:


# Calculate TPR and FPR for each level of confidence
for confidence_level in confidence_levels:
    true_positive_hits = subset[subset['Confidence'] >= confidence_level]['mean_hits_list'].sum()
    false_negative_misses = subset[subset['Confidence'] >= confidence_level]['mean_misses_list'].sum()
    false_positive_fas = subset[subset['Confidence'] >= confidence_level]['mean_fas_div_list'].sum()
    true_negative_correct_rej = subset[subset['Confidence'] >= confidence_level]['mean_correct_rej_list'].sum()


# In[396]:


if true_positive_hits + false_negative_misses == 0:
    tpr = 1.0
else:
    tpr = true_positive_hits / (true_positive_hits + false_negative_misses)

if false_positive_fas + true_negative_correct_rej == 0:
    fpr = 0.0
else:
    fpr = false_positive_fas / (false_positive_fas + true_negative_correct_rej)


# In[397]:


tpr_list.append(tpr)
fpr_list.append(fpr)


# In[398]:


lin['true_labels'] = None

lin.loc[(lin['Decision'] == lin['correct_answer']) & (lin['target_presence'] == 'tp'), 'true_labels'] = 1
lin.loc[(lin['Decision'] != 'Reject') & (lin['correct_answer'] == 'Reject'), 'true_labels'] = 0
lin['true_labels'].fillna(2, inplace=True)  # Fill NaN values with 2


# In[399]:


true_labels = lin['true_labels']
confidence = lin['Confidence']

fpr, tpr, _ = roc_curve(true_labels, confidence, pos_label=1)

roc_auc = auc(fpr, tpr)

import pandas as pd
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

plt.plot(fpr, tpr, label=f'{dist_level}, {del_level} (AUC = {roc_auc:.2f})')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()


# In[ ]:




