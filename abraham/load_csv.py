import csv
import os

# function to load a CSV file into a dictionary
# the dictionary keys are the column headers and the values are lists of column data
def load_csv_to_dict(file_path, name):
    with open(file_path, mode='r', encoding='utf-8') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Initialize a dictionary with empty lists for each column header
        data_dict = {field: [] for field in csv_reader.fieldnames}
        data_dict['name'] = name  # Add the experiment name to the dictionary

        # Populate the dictionary with data from the CSV file
        for row in csv_reader:
            for field in csv_reader.fieldnames:
                data_dict[field].append(row[field])

    return data_dict

# function tofind all results.csv files
# returns a list of tuples containing the parent directory name and the path to the results.csv file
def load_results_csv_paths(root_dir):
    results_list = []

    # Walk through the directory structure starting from root_dir
    for root, dirs, files in os.walk(root_dir):
        for dir_name in dirs:
            # Construct the path to the CSV directory
            csv_dir_path = os.path.join(root, dir_name, 'csv')
            results_path = os.path.join(csv_dir_path, 'results.csv')
            
            # Check if results.csv exists in the constructed path
            if os.path.isfile(results_path):
                # Extract the parent directory name one level higher
                parent_dir = os.path.basename(root)
                results_list.append((parent_dir, results_path))

    return results_list

# set the root directory to start searching for results.csv files, should be the folder containing the results folder
root_directory = '.'
results_list = load_results_csv_paths(root_directory)

csv_list = []

# load each results.csv file into a dictionary and add it to csv_list
for name, path in results_list:
    results = load_csv_to_dict(path, name)
    csv_list.append(results)

# now csv_list contains a list of dictionaries, each dictionary representing a CSV file
# the fields in each dictionary are 'name' (experiment description) and the CSV headers

# example print statements to show some data from the frist csv dictionary in csv_list
print("Experiment name:", csv_list[0]['name'])
print("Number of Ground Truth Answers:", len(csv_list[0]['Ground Truth Answer']))
print("First GPT Response:", csv_list[0]['GPT Response'][0])
print("First Question ID:", csv_list[0]['Question ID'][0])
print("First Prompt:", csv_list[0]['Prompt'][0])