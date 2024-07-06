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