import pandas as pd
from pathlib import Path
import os

# Base path of parent folder
base_path = "/Users/beckercheng/Desktop/ECE 4905/emg-haptic-prototype/EMG_data_for_gestures-master"

# Convert to Path object
base_path = Path(base_path).expanduser()

# Make sure path exists
if not base_path.exists():
    print(f"Path not found: {base_path}")
    print("Please update the base_path variable with the correct path")
else:
    # Process all subject folders
    for i in range(1, 10):
        folder_path = base_path / ('0'+str(i))
        
        if folder_path.exists():
            # Get all txt files
            for txt_file in folder_path.glob("*.txt"):
                print(f"Converting {txt_file.name} in folder {i}")
                
                # Read and convert
                try:
                    df = pd.read_csv(txt_file, sep='\s+', engine='python')

                    # Save as CSV
                    csv_file = txt_file.with_suffix('.csv')
                    df.to_csv(csv_file, index=False)
                    print(f"Successfully converted {txt_file.name}")

                    # Delete original txt file
                    txt_file.unlink()

                except Exception as e:
                    print(f"Failed to convert {txt_file.name}: {e}")
                
    print("Conversion complete!")