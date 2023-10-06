# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# + tags=[]
import os
import datetime
import shutil


# + tags=[]
def backup_logger_file(logger_version_path):
    
    metric_file = os.path.join(logger_version_path, 'metrics.csv')
    m_time = os.path.getmtime(metric_file)
    m_time = datetime.datetime.fromtimestamp(m_time)
    m_time = m_time.strftime('%Y-%m-%d-%H:%M:%S')

    if os.path.exists(metric_file):
        backup_file = metric_file.replace('.csv', f'-{m_time}.csv')
        if not os.path.exists(backup_file):
            shutil.copy2(metric_file, backup_file)

# + tags=["active-ipynb", "style-student"]
# path = '/usr/local/ay_data/1-model_save/3-CS/CSNet+/coco/1/version_0'
# backup_logger_file(path)
