import pandas as pd
from auxiliar_functions.preprocessing_functions import preprocess_sotab_split

def main():
    source_folder = 'data/raw_data'
    save_path = 'data/structured_data/'

    # Preprocess the Train directory
    print('Preprocessing the Train directory')
    train_directory = source_folder + '/Train/'
    train_csv_file = source_folder + '/CTA_training_gt.csv'
    train_df = preprocess_sotab_split(train_directory, train_csv_file)

    # Preprocess the Test directory
    print('Preprocessing the Test directory')
    test_directory = source_folder + '/Test/'
    test_csv_file = source_folder + '/CTA_test_gt.csv'
    test_df = preprocess_sotab_split(test_directory, test_csv_file)

    # Preprocess the Validation directory
    print('Preprocessing the Validation directory')
    validation_directory = source_folder + '/Validation/'
    validation_csv_file = source_folder + '/CTA_validation_gt.csv'
    dev_df = preprocess_sotab_split(validation_directory, validation_csv_file)

    # Create a dictionary to store the DataFrames
    dataframes = {'train': train_df, 'dev': dev_df, 'test': test_df}

    # Specify the path where you want to save the pickle file
    pickle_file_path = save_path+'sotab_data_preprocessed.pkl'

    # Save the dictionary of DataFrames to a pickle file
    with open(pickle_file_path, 'wb') as file:
        pd.to_pickle(dataframes, file)

if __name__ == '__main__':
    main()