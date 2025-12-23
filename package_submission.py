import os
import zipfile
import shutil
from datetime import datetime

# Configuration
PROJECT_ROOT = os.getcwd()
OUTPUT_FILENAME = "Indecimal_RAG_Submission.zip"
EXCLUDE_DIRS = {'node_modules', 'venv', '__pycache__', '.git', '.gemini', 'dist', 'coverage'}
EXCLUDE_EXTENSIONS = {'.pyc', '.DS_Store'}

def zip_project(source_dir, output_filename):
    print(f"📦 Packaging project from: {source_dir}")
    print(f"🚫 Excluding: {', '.join(EXCLUDE_DIRS)}")
    
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, dirs, files in os.walk(source_dir):
            # Modify dirs in-place to skip excluded directories
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            
            for file in files:
                if any(file.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
                    continue
                
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, source_dir)
                
                # print(f"  Adding: {arcname}")
                zipf.write(file_path, arcname)
                
    print(f"\n✅ Ready! Submission file created: {output_filename}")
    print(f"📊 Size: {os.path.getsize(output_filename) / 1024 / 1024:.2f} MB")

if __name__ == "__main__":
    zip_project(PROJECT_ROOT, OUTPUT_FILENAME)
