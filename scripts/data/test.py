import pandas as pd

# Load the dataset
file_path = 'merged_data.csv' # Update this path to your dataset
data = pd.read_csv(file_path)

# Extract unique flat types
flat_type_column = 'flat_model'  # Update this to the correct column name
unique_flat_types = [flat_model.lower().strip() for flat_model in data[flat_type_column].unique()]


# Print the unique flat types
print("Unique flat models:")
for flat_model in unique_flat_types:
  print(flat_model)
print(len(unique_flat_types))