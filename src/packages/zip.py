import os
import zipfile

def zip_directories(output_zip, *directories):
    with zipfile.ZipFile(output_zip, 'w') as zipf:
        for directory in directories:
            for root, _, files in os.walk(directory):
                for file in files:
                    file_path = os.path.join(root, file)
                    arcname = os.path.relpath(file_path, os.path.join(directory, '..'))
                    zipf.write(file_path, arcname)
                    
# Function to zip folders considering exclusion list
def zip_dir(dir_to_zip, output_zip, exclude=[]):
    exclude = [os.path.abspath(os.path.join(dir_to_zip, ex_folder)) for ex_folder in exclude]
    with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
        # Iterate over all files and folders in the directory
        for root, _, files in os.walk(dir_to_zip):
            abs_root = os.path.abspath(root)
            # Check if the current root is in the exclude list
            if any(abs_root.startswith(ex_folder) for ex_folder in exclude):
                continue  # Skip this folder and its contents if in exclude list

            for file in files:
                if file == os.path.basename(output_zip):
                    continue  # Skip the output zip file itself
                abs_file = os.path.join(root, file)
                zipf.write(abs_file, os.path.relpath(abs_file, dir_to_zip))


# Function to list folders within a zip file
def list_folders_and_files_in_zip(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zipf:
        file_set = set()

        # Iterate over each entry in the zip file
        for entry in zipf.infolist():
            entry_name = entry.filename

            # Determine if entry is a directory or file
            if entry_name.endswith('/'):
                # Entry is a directory
                folder_name = entry_name.rstrip('/')
            else:
                # Entry is a file
                file_set.add(entry_name)

        print("Files in the zip file:")
        for file in file_set:
            print(file)
