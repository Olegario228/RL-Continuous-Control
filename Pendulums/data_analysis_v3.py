import pandas as pd

# Function to load and preprocess the data
def preprocess_data(file_path):
    cleaned_lines = []
    with open(file_path, 'r') as file:
        for line in file:
            if not line.startswith('#'):
                cleaned_line = line.replace('(', '').replace(')', '').strip()
                cleaned_lines.append(cleaned_line)
    return pd.DataFrame([line.split(',') for line in cleaned_lines if line])

def convert_columns_to_numeric(df):
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except ValueError:
            pass  # keep as string if conversion fails
    return df

# Function to extract and modify the header
def get_modified_header(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#globalIndex'):
                header = line.strip('#\n').split(',')
                # Replace 'value' with 'MaxSuccessfulTests' and 'MaxSuccessfulTestsTimesteps'
                header[1] = 'MaxSuccessfulTests'
                header.insert(2, 'MaxSuccessfulTestsTimesteps')
                return ','.join(header)
    return ''

# Your original file path
file_path = 'models\Curriculum-Learning\PPO_CL_Var3_X5_ResultsVar.txt'

# Extract the key part from the filename and get the modified header
key_part = file_path.split('/')[-1].split('_')[2]

# Or for the decay types
# Splitting the filename from the path and then splitting by underscores
filename_parts = file_path.split('\\')[-1].split('_')

# Finding the index of the key part (e.g., Var3)
key_part_index = [i for i, part in enumerate(filename_parts) if 'Var' in part]

# Ensuring that there is a part after the key part and extracting it
if key_part_index and len(filename_parts) > key_part_index[0] + 1:
    key_part = filename_parts[key_part_index[0]]
    dynamic_part = filename_parts[key_part_index[0] + 1]
else:
    key_part = None
    dynamic_part = None

modified_header = get_modified_header(file_path) + '\n'

# Preprocessing the data
data = preprocess_data(file_path)
data = convert_columns_to_numeric(data)

# Define the condition for successful tests and thresholds
successful_tests_threshold = 45
thresholds = [17000, 15000, 13000]
lower_limit = 10000
upper_limit = float('inf')

# Filter the results and create a csv to work with
# filtered_data = data[data[1] > successful_tests_threshold]
filtered_data = data # UNFILTERED, for general plotting
output_file = f'{key_part}_{dynamic_part}_results.csv'
with open(output_file, 'w', newline='') as file:
        file.write(modified_header)
        filtered_data.to_csv(file, index=False, header=False)
print(f'Results saved into: {output_file}')

# # Filtering data and saving to CSV with the modified header
# for i in range(len(thresholds)):
#     upper_limit = thresholds[i]
#     lower_bound = lower_limit if i == len(thresholds) - 1 else thresholds[i + 1]
    
#     # filtering and showing
#     filtered_data = data[(data[2] < upper_limit) & (data[2] >= lower_bound) & (data[1] > successful_tests_threshold)]
#     count = filtered_data.shape[0]
#     print(f'Number of tests between {lower_bound} and {upper_limit} with successful tests > {successful_tests_threshold}: {count}')
    
#     # create the final CSV file and prepend the modified header
#     output_file = f'{key_part}_in_timesteps_[{lower_bound}-{upper_limit}].csv'
#     with open(output_file, 'w', newline='') as file:
#         file.write(modified_header)
#         filtered_data.to_csv(file, index=False, header=False)
    
#     print(f'Data saved to {output_file}')
