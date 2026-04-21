# SP26 ECE 4905 Capstone Team 12 AKLO

EMG Dataset: Online EMG dataset used as the reference for rest vs. fist muscle activity.

Threshold Calculation Rest_Fist: Processes the online dataset to compute a threshold that separates fist (active) and rest (inactive) signals.

Simulated Test Files: Generated test EMG files based on the online dataset to represent Morse code input sequences.

Morse Code Translation: Imports simulated test files and outputs Morse code (dots/dashes) and corresponding alphanumeric characters.

Raspberry Pi Code: Processes raw EMG signals on the Pi and controls haptic motors based on decoded Morse output.
