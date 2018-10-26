
# coding: utf-8

# In[48]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import pickle


# In[37]:


def get_csv_list(dir_path):
    csv_path_list = glob.glob(dir_path+'*.csv')
    
    return csv_path_list


# In[39]:


def create_merged_dataframe(csv_path_list):
    df_all = pd.DataFrame()
    df_all['filename'] = pd.read_csv(csv_path_list[0])['filename']
    df_all['score'] = [0 for _ in range(len(df_all))]
    df_all['num_of_evaluations'] = [0 for _ in range(len(df_all))]
    df_all.sort_values(by=['filename'], inplace=True)

    for csv_path in csv_path_list:
        df_tmp = pd.read_csv(csv_path)
        df_tmp.sort_values(by=['filename'], inplace=True)
        df_all['score'] = df_all['score'] + df_tmp['score']
        df_all['num_of_evaluations'] = df_all['num_of_evaluations'] + df_tmp['num_of_evaluations']

    df_all['mean_score'] = df_all['score'] / df_all['num_of_evaluations']
    
    return df_all


# In[46]:


def get_file_score_list_from_all_data(df_all):
    df_result = pd.read_csv(latest_df_path)
    file_list = df_result['filename']
    score_list = (df_result['score'] / df_result['num_of_evaluations']) // 0.2 + 1.0
    
    return file_list, score_list


# In[51]:


def save_file_score_list(file_list, score_list):
    try:
        with open('./pickle/file_list.pickle', 'wb') as f:
            pickle.dump(file_list, f)
        with open('./pickle/score_list.pickle', 'wb') as f:
            pickle.dump(score_list, f)
    except:
        raise


# In[50]:


def load_file_score_list():
    try:
        with open('./pickle/file_list.pickle', 'rb') as f:
            file_list = pickle.load(f)
        with open('./pickle/score_list.pickle', 'rb') as f:
            scorelist = pickle.load(f)
    except:
        raise
        
    return file_list, score_list


# In[52]:


def main():
    DIR_PATH = './data/all_apart_data/'
    csv_list = get_csv_list(DIR_PATH)
    df_all = create_merged_dataframe(csv_list)
    file_list, score_list = get_file_score_list_from_all_data(df_all)
    save_file_score_list(file_list, score_list)


# In[53]:


if __name__=='__main__':
    main()

