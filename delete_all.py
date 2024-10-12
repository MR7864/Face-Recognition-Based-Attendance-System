import os

# Paths to your pickle files
names_pickle_path = 'data/names.pkl'
faces_data_pickle_path = 'data/faces_data.pkl'

# Delete the pickle files if they exist
if os.path.exists(names_pickle_path):
    os.remove(names_pickle_path)

if os.path.exists(faces_data_pickle_path):
    os.remove(faces_data_pickle_path)

# Confirm deletion
print("Dataset cleared. Ready for a new dataset.")
