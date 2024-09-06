import os

def delete_pth_files(folder_path):
    """
    Delete all .pth files in the specified directory, including its subdirectories.
    """
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(".pth") and not ("policy" in file):
                os.remove(os.path.join(root, file))
                print(f"Deleted: {os.path.join(root, file)}")

# Example usage (you would replace 'your_folder_path' with the actual folder path):
# delete_pth_files('your_folder_path')

delete_pth_files("/home/mrudolph/documents/actbisim/scripts_nav2d/tianshou_rl/log")