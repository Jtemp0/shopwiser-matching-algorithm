import os

def dump_codebase_to_txt(target_directory, output_file="project_dump.txt"):
    if not os.path.exists(target_directory):
        print(f"Error: The path '{target_directory}' does not exist.")
        return

    # List of folder names to strictly ignore
    excluded_folders = {'__pycache__', '.git', '.venv', 'node_modules'}

    with open(output_file, 'w', encoding='utf-8') as out:
        out.write("### FOLDER STRUCTURE ###\n")
        out.write("-" * 25 + "\n")
        
        # 1. Write the folder structure tree
        for root, dirs, files in os.walk(target_directory):
            # Modify dirs in-place to exclude unwanted folders
            dirs[:] = [d for d in dirs if d not in excluded_folders]
            
            level = root.replace(target_directory, '').count(os.sep)
            indent = ' ' * 4 * level
            folder_name = os.path.basename(root) if os.path.basename(root) else target_directory
            out.write(f"{indent}{folder_name}/\n")
            
            sub_indent = ' ' * 4 * (level + 1)
            for f in files:
                if f != output_file:
                    out.write(f"{sub_indent}{f}\n")

        out.write("\n" + "="*60 + "\n")
        out.write("### FILE CONTENTS ###\n")
        out.write("-" * 25 + "\n")

        # 2. Iterate to read and write file contents
        for root, dirs, files in os.walk(target_directory):
            # Ensure consistency by excluding the same folders here
            dirs[:] = [d for d in dirs if d not in excluded_folders]
            
            for file in files:
                if file == output_file: 
                    continue
                    
                file_path = os.path.join(root, file)
                relative_path = os.path.relpath(file_path, target_directory)
                
                out.write(f"\nFILE: {relative_path}\n")
                out.write("-" * (len(relative_path) + 6) + "\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        out.write(f.read())
                except Exception as e:
                    out.write(f"[Skipped non-text or unreadable file: {file}]")
                
                out.write("\n\n" + "~" * 40 + "\n")

    print(f"Success! Clean dump (excluding pycache) saved to: {output_file}")

if __name__ == "__main__":
    # You can specify a full path here, e.g., r'C:\Users\Mihail\Project'
    dump_codebase_to_txt('src/shopwiser')