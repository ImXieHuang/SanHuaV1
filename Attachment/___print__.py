import os
import sys

def generate_file_tree(start_path, output_file, filtered_dirs=None):
    if filtered_dirs is None:
        filtered_dirs = []
    
    def write_tree(directory, prefix="", is_last=True):
        connector = "└── " if is_last else "├── "
        f.write(prefix + connector + os.path.basename(directory) + "\n")
        
        new_prefix = prefix + ("    " if is_last else "│   ")
        
        try:
            items = []
            for item in os.listdir(directory):
                item_path = os.path.join(directory, item)
                if os.path.isdir(item_path):
                    items.append((item, True))
                else:
                    items.append((item, False))
            
            items.sort(key=lambda x: (not x[1], x[0]))
            
            for i, (item, is_dir) in enumerate(items):
                item_path = os.path.join(directory, item)
                is_last_item = (i == len(items) - 1)
                
                if is_dir:
                    if item in filtered_dirs:
                        connector = "└── " if is_last_item else "├── "
                        f.write(new_prefix + connector + item + " [Permission denial: ...]\n")
                    else:
                        write_tree(item_path, new_prefix, is_last_item)
                else:
                    connector = "└── " if is_last_item else "├── "
                    f.write(new_prefix + connector + item + "\n")
                    
        except PermissionError:
            f.write(new_prefix + "└── [Permission denial]\n")
    
    def write_file_content(file_path, relative_path):
        try:
            if file_path.endswith('.py'):
                f.write(f"\n# {relative_path}:\n")
                f.write('"""\n')
                
                with open(file_path, 'r', encoding='utf-8') as file:
                    content = file.read()
                    f.write(content)
                    if content and not content.endswith('\n'):
                        f.write('\n')
                
                f.write('"""\n')
                
        except UnicodeDecodeError:
            f.write(f"# {relative_path}:\n")
            f.write('"""\n')
            f.write("# [Error: File content cannot be read due to encoding issues]\n")
            f.write('"""\n')
        except Exception as e:
            f.write(f"# {relative_path}:\n")
            f.write('"""\n')
            f.write(f"# [Error: File content cannot be read - {str(e)}]\n")
            f.write('"""\n')
    
    def process_directory(directory):
        write_tree(directory)
        
        f.write("\n" + "="*80 + "\n")
        f.write("File content:\n")
        f.write("="*80 + "\n")
        
        for root, dirs, files in os.walk(directory):
            dirs[:] = [d for d in dirs if not d.startswith('.') and d not in filtered_dirs]
            
            for file in files:
                if file.endswith('.py'):
                    file_path = os.path.join(root, file)
                    relative_path = os.path.relpath(file_path, start_path)
                    write_file_content(file_path, relative_path)
    
    if not os.path.exists(start_path):
        print(f"Error: Path '{start_path}' doesn't exist")
        return
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("Project file tree:\n")
        f.write("="*80 + "\n")
        
        process_directory(start_path)
        
        print(f"The file tree and content have been successfully output to: {output_file}")

def main():
    if len(sys.argv) > 1:
        project_path = sys.argv[1]
    else:
        project_path = "."
    
    if len(sys.argv) > 2:
        output_filename = sys.argv[2]
    else:
        output_filename = "Attachment\\EngineeringDocument.txt"
    
    filtered_dirs = ['.git', 'Attachment']
    if len(sys.argv) > 3:
        filtered_dirs = sys.argv[3].split(',')
    
    if not output_filename.endswith('.txt'):
        output_filename += '.txt'
    
    print(f"Scanning the project: {os.path.abspath(project_path)}")
    print(f"Output file: {output_filename}")
    print(f"Filtering directory: {filtered_dirs}")
    
    generate_file_tree(project_path, output_filename, filtered_dirs)

if __name__ == "__main__":
    main()