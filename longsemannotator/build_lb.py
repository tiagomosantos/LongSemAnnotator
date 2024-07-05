from sklearn.preprocessing import LabelBinarizer
import pandas as pd
from auxiliar_functions.general_purpose_functions import save_to_pickle

with open('data/structured_data/sotab_data_preprocessed.pkl', 'rb') as file:
    df = pd.read_pickle(file)

df_train = df['train']
df_test = df['test']
df_dev = df['dev']

cta_labels = df['train']['label'].unique()

cta_labels_as_list = [label for label in cta_labels]

lb = LabelBinarizer()
lb.fit(cta_labels_as_list)

save_to_pickle(lb,'data/ready_to_model_data/sotab_lb.pkl')