import pandas as pd
from pathlib import Path
import os

### **What this creates:**

#1. **test_all_rest.csv**: 1000 continuous rest samples (5 windows × 200 samples)
#2. **test_all_fist.csv**: 1000 continuous fist samples (5 windows × 200 samples)
#3. **test_alternating.csv**: R-F-R-F-R pattern (200 samples each)

#test_alternating.csv structure:
#amples 0-199:    Rest  (Window 1)
#Samples 200-399:  Fist  (Window 2)
#amples 400-599:  Rest  (Window 3)
#Samples 600-799:  Fist  (Window 4)
#Samples 800-999:  Rest  (Window 5)

# Simple direct approach
base_path = Path("EMG_data_for_gestures-master_csv") / "33"

# Create output folder for test files
output_folder = Path("test_files")
output_folder.mkdir(exist_ok=True)  # Creates folder if it doesn't exist

# Get first CSV file
first_file = list(base_path.glob("*.csv"))[0]
data = pd.read_csv(first_file)

# Separate classes
rest = data[data['class'] == 1]
fist = data[data['class'] == 2]

# Create test files in the test_files folder
# 1. All rest (1000 samples)
rest[:1000].to_csv(output_folder / 'test_all_rest.csv', index=False)

# 2. All fist (1000 samples)
fist[:1000].to_csv(output_folder / 'test_all_fist.csv', index=False)

# 3. Alternating pattern (200 samples each)
alternating = pd.concat([
    rest[0:200],      # Window 1: Rest
    fist[0:200],      # Window 2: Fist
    rest[200:400],    # Window 3: Rest
    fist[200:400],    # Window 4: Fist
    rest[400:600]     # Window 5: Rest
], ignore_index=True)

alternating.to_csv(output_folder / 'test_alternating.csv', index=False)

print(f"Files created in '{output_folder}' folder!")
print(f"  test_all_rest.csv: {len(rest[:1000])} samples")
print(f"  test_all_fist.csv: {len(fist[:1000])} samples")
print(f"  test_alternating.csv: {len(alternating)} samples")

# List all files in the output folder to verify
print(f"\nFiles in {output_folder}:")
for file in output_folder.glob("*.csv"):
    print(f"  - {file.name}")