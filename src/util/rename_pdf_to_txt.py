import os

def rename_files(base_dir, old_ext=".PDF", new_ext=".txt"):
    for root, _, files in os.walk(base_dir):
        for file in files:
            if file.endswith(old_ext):
                full_path = os.path.join(root, file)
                
                new_name = full_path.rsplit(old_ext, 1)[0] + new_ext
                os.rename(full_path, new_name)
                print(f"Renamed: {full_path} -> {new_name}")

# Replace with the path to your directory
base_directory = "/home/bowserj/profunc/data/text_output"
rename_files(base_directory)

