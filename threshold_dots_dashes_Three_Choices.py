"""
EMG Morse Code Detector
-----------------------
Detects dots and dashes from EMG signals using MAV threshold
and duration-based classification.

Input: CSV file with EMG data (1000Hz sampling rate)
Output: Sequence of more code dots (.) and dashes (-)
"""

import pandas as pd
import numpy as np

# =============================================================================
# CONFIGURABLE PARAMETERS
# =============================================================================

# EMG threshold for detecting fist vs rest (in volts)
THRESHOLD = 0.00005022  # Calculated from CDR

# Sampling parameters
SAMPLING_RATE = 1000  # Hz
WINDOW_SIZE = 200     # samples (200ms at 1000Hz)
OVERLAP = 0.5         # 50% overlap

# Calculated: time between consecutive MAV values
STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))  # 100 samples = 100ms
MAV_INTERVAL = STEP_SIZE / SAMPLING_RATE       # 0.1 seconds between MAV values

# Duration thresholds (in seconds)
MIN_ACTIVATION = 0.5    # minimum duration to count as valid input (filters noise)
DASH_THRESHOLD = 2.0    # >= this duration = DASH, < this duration = DOT

# Gap thresholds (in seconds)
MIN_RELEASE = 0.5       # minimum rest between dots and dashes within a letter
LETTER_GAP = 3.0        # rest duration to finalize a letter
WORD_GAP = 6.0          # rest duration to finalize a word (insert space)

# Convert time thresholds to sample counts (MAV samples, not raw samples)
MIN_ACTIVATION_SAMPLES = int(MIN_ACTIVATION / MAV_INTERVAL)
DASH_THRESHOLD_SAMPLES = int(DASH_THRESHOLD / MAV_INTERVAL)
MIN_RELEASE_SAMPLES = int(MIN_RELEASE / MAV_INTERVAL)
LETTER_GAP_SAMPLES = int(LETTER_GAP / MAV_INTERVAL)
WORD_GAP_SAMPLES = int(WORD_GAP / MAV_INTERVAL)

# Channel names (modify this list based on your CSV column names)
CHANNEL_PREFIX = "channel"

# =============================================================================
# MAV CALCULATION
# =============================================================================

def calculate_mav(channel_data):
    """
    Take raw EMG data from multiple channels, calculate MAV for each channel, 
    then average across all channels.
    
    Args:
        channel_data: dict of {channel_name: 1D numpy array} 
        (Processed while loading in csv file)
        
    Returns:
        averaged_mav: 1D numpy array of averaged MAV values (per window) across all channels
    """
    channel_mavs = [] # Use to hold MAV array for each channel
    
    # ch_name is the key, signal is value (loop for channel number times)
    for ch_name, signal in channel_data.items():
        mav_values = []
        start = 0
        
        # Keep looping while we have enough samples for a full window
        while start + WINDOW_SIZE <= len(signal):
            window = signal[start:start + WINDOW_SIZE]
            mav = np.mean(np.abs(window))
            mav_values.append(mav)
            start += STEP_SIZE
        
        # Store the window MAV for each channel
        channel_mavs.append(mav_values)
    
    # Stack all channel MAVs and average across channels
    all_mavs = np.array(channel_mavs)  # [num_channels rows, num_windows columns]
    averaged_mav = np.mean(all_mavs, axis=0)  # Row based mean, [num_windows,]
    
    return averaged_mav

# =============================================================================
# MORSE DETECTION STATE MACHINE (Dot and Dash Classification)
# =============================================================================

class MorseDetector:
    """
    State machine for detecting Morse code elements from MAV values.
    
    States:
        IDLE: Waiting for activation (fist clench)
        ACTIVE: Currently detecting a fist clench
        GAP: Fist released, counting gap duration
    """

    # Constructor (Variables)
    def __init__(self):
        self.state = "IDLE"        # Idle, active, or gap
        self.active_count = 0      # consecutive samples above threshold
        self.gap_count = 0         # consecutive samples below threshold
        self.current_letter = []   # buffer for current letter elements
        self.sequence = []         # Array that saves the output sequence of dots and dashes
        self.output = []           # final output (dots, dashes, spaces)
        
    def process_mav(self, mav_value):
        """
        Process a single MAV value and update state.
        
        Args:
            mav_value: single MAV measurement
            
        Returns:
            event: string describing what happened, or None
        """
        
        # Fist Gesture detected if MAV is above threshold, otherwise it's a rest
        is_active = mav_value > THRESHOLD 
        event = None
        
        if self.state == "IDLE":
            if is_active:
                # Start of potential activation
                self.state = "ACTIVE"
                self.active_count = 1
                
        elif self.state == "ACTIVE":
            if is_active:
                # Continue counting activation duration
                self.active_count += 1
            else:
                # Activation ended, classify as dot or dash
                event = self._classify_activation()  # Use to classify dot or dash
                self.state = "GAP"
                self.gap_count = 1
                self.active_count = 0
                
        # Under 0.5 seconds means noise (ignore it)
        # 0.5 to 3.0 seconds means a valid pause between dots/dashes within the same letter
        # 3.0 seconds triggers a letter boundary
        # 6.0 seconds triggers a word boundary.
        elif self.state == "GAP":
            if is_active:
                # Reactivation during gap, could be noise or start of next activation
                if self.gap_count >= MIN_RELEASE_SAMPLES:
                    self.state = "ACTIVE"
                    self.active_count = 1
                    self.gap_count = 0
                # else: ignore it, keep counting the gap
                else:
                    self.gap_count += 1
            else: 
                self.gap_count += 1
                
                # Check for letter/word boundaries
                if self.gap_count == LETTER_GAP_SAMPLES:
                    event = self._finalize_letter()
                elif self.gap_count == WORD_GAP_SAMPLES:
                    event = self._finalize_word()
    
        # Return event (dot, dash, noise (too short), letter complete, word boundary) 
        # as a string for logging, or None if no significant event
        # string for logging, or None if no significant event
        return event 
    
    def _classify_activation(self):
        """Classify activation duration as DOT, DASH, or noise."""
        if self.active_count >= DASH_THRESHOLD_SAMPLES:
            self.current_letter.append("-")
            self.sequence.append("-") 
            return "DASH"
        elif self.active_count >= MIN_ACTIVATION_SAMPLES:
            self.current_letter.append(".")
            self.sequence.append(".") 
            return "DOT"
        else:
            return "NOISE (too short)"
    
    def _finalize_letter(self):
        """Finalize current letter and add to output."""
        if self.current_letter:
            letter = "".join(self.current_letter)
            self.output.append(letter)
            self.sequence.append("|") 
            self.current_letter = []
            return f"LETTER COMPLETE: {letter}"
        return None
    
    def _finalize_word(self):
        """Add word boundary (space) to output."""
        if self.output and self.output[-1] != " ":
            self.output.append(" ")
            # Replace the last "|" with "||" for word boundary
            if self.sequence and self.sequence[-1] == "|":
                self.sequence[-1] = "||"
            return "WORD BOUNDARY"
        return None
    
    def finalize(self):
        """Call at end of signal to finalize any remaining content."""
        if self.current_letter:
            letter = "".join(self.current_letter)
            self.output.append(letter)
            self.current_letter = []
    
    def get_output(self):
        """Get the full output sequence."""
        return self.output
    
    def get_output_string(self):
        """Get output as a formatted string."""
        return " | ".join(self.output)
    
    def get_sequence(self):
        return self.sequence
    
    def get_sequence_string(self):
        return "".join(self.sequence)

# =============================================================================
# MAIN PROCESSING FUNCTION
# =============================================================================

def process_emg_file(filepath):
    """
    Process an EMG CSV file with multiple channels and detect Morse code elements.
    Automatically finds columns that start with the channel prefix.
    
    Args:
        filepath: path to CSV file
        
    Returns:
        detector: MorseDetector object with results
        averaged_mav: array of averaged MAV values across channels
    """

    # Load data
    df = pd.read_csv(filepath)
    
    # Auto-detect channel columns (columns starting with prefix)
    channel_columns = [col for col in df.columns if col.startswith(CHANNEL_PREFIX)]
    
    if len(channel_columns) == 0:
        available = list(df.columns)
        raise ValueError(f"No columns found starting with '{CHANNEL_PREFIX}'\nAvailable columns: {available}")
    
    # Sort to ensure consistent order (channel1, channel2, ... channel10, etc.)
    channel_columns = sorted(channel_columns, key=lambda x: int(x.replace(CHANNEL_PREFIX, "") or 0))
    
    print(f"Found {len(channel_columns)} channels: {channel_columns}")
    
    # Extract channel data into Python dictionary, use for import to calculate MAV
    channel_data = {col: df[col].values for col in channel_columns}
    
    num_samples = len(df)
    print(f"Loaded {num_samples} samples ({num_samples/SAMPLING_RATE:.2f} seconds)")
    
    # Calculate MAV averaged across all channels
    averaged_mav = calculate_mav(channel_data)
    print(f"Calculated {len(averaged_mav)} MAV values (averaged across {len(channel_columns)} channels)")
    
    # Process through detector
    detector = MorseDetector()
    
    print("\n--- Processing ---")
    # Loop through all windows (or all MAVs)
    for i, mav in enumerate(averaged_mav):
        event = detector.process_mav(mav)
        if event:
            time_sec = i * MAV_INTERVAL
            print(f"[{time_sec:6.2f}s] {event}")
    
    # Finalize any remaining content
    detector.finalize()
    
    return detector, averaged_mav

# =============================================================================
# Print Parameters Configuration
# =============================================================================

def print_config():
    """Print current configuration."""
    print("=" * 50)
    print("EMG MORSE DETECTOR CONFIGURATION")
    print("=" * 50)
    print(f"Threshold:          {THRESHOLD} V")
    print(f"Sampling rate:      {SAMPLING_RATE} Hz")
    print(f"Window size:        {WINDOW_SIZE} samples ({WINDOW_SIZE/SAMPLING_RATE*1000:.0f}ms)")
    print(f"Overlap:            {OVERLAP*100:.0f}%")
    print(f"MAV interval:       {MAV_INTERVAL*1000:.0f}ms")
    print()
    print(f"Channel prefix:     '{CHANNEL_PREFIX}' (auto-detects channel1, channel2, ...)")
    print()
    print("Duration thresholds:")
    print(f"  Min activation:   {MIN_ACTIVATION}s ({MIN_ACTIVATION_SAMPLES} samples)")
    print(f"  Dash threshold:   {DASH_THRESHOLD}s ({DASH_THRESHOLD_SAMPLES} samples)")
    print()
    print("Gap thresholds:")
    print(f"  Min release:      {MIN_RELEASE}s ({MIN_RELEASE_SAMPLES} samples)")
    print(f"  Letter gap:       {LETTER_GAP}s ({LETTER_GAP_SAMPLES} samples)")
    print(f"  Word gap:         {WORD_GAP}s ({WORD_GAP_SAMPLES} samples)")
    print("=" * 50)

# =============================================================================
# Morse Code Dots and Dashes Mapping
# =============================================================================

def translate_morse(morse_array):
    translation = ""
    current_letter = ""
    i = 0

    dictionary = {
        ".-":"a", "-...":"b", "-.-.":"c", "-..":"d",
        ".":"e", "..-.":"f", "--.":"g", "....":"h",
        "..":"i", ".---":"j", "-.-":"k", ".-..":"l",
        "--":"m", "-.":"n", "---":"o", ".--.":"p",
        "--.-":"q", ".-.":"r", "...":"s", "-":"t",
        "..-":"u", "...-":"v", ".--":"w", "-..-":"x",
        "-.--":"y", "--..":"z", ".----": "1", "..---":"2",
        "...--":"3", "....-":"4", ".....":"5", "-....":"6",
        "--...":"7", "---..":"8", "----.":"9", "-----":"0"
    }

    while i < len(morse_array):

        if morse_array[i] == "||":  # Word boundary
            translation += dictionary.get(current_letter, "?")
            translation += " "
            current_letter = ""

        elif morse_array[i] == "|":  # Letter boundary
            translation += dictionary.get(current_letter, "?")
            current_letter = ""

        else:  # Dot or dash
            current_letter = current_letter + morse_array[i]

        i += 1

    # Add final letter if exists
    if current_letter:
        translation += dictionary.get(current_letter, "?")

    return translation

# =============================================================================
# TEXT TO MORSE (Custom Input)
# =============================================================================

TEXT_TO_MORSE = {
    "a": ".-",   "b": "-...", "c": "-.-.", "d": "-..",
    "e": ".",    "f": "..-.", "g": "--.",  "h": "....",
    "i": "..",   "j": ".---", "k": "-.-", "l": ".-..",
    "m": "--",   "n": "-.",   "o": "---", "p": ".--.",
    "q": "--.-", "r": ".-.",  "s": "...", "t": "-",
    "u": "..-",  "v": "...-", "w": ".--", "x": "-..-",
    "y": "-.--", "z": "--..",
    "1": ".----", "2": "..---", "3": "...--", "4": "....-",
    "5": ".....", "6": "-....", "7": "--...", "8": "---..",
    "9": "----.", "0": "-----"
}

def text_to_morse_array(text):
    morse_array = []
    skipped = []
    words = text.lower().split(" ")

    for word_index, word in enumerate(words):
        if not word:
            continue
        for letter_index, char in enumerate(word):
            morse = TEXT_TO_MORSE.get(char)
            if morse is None:
                skipped.append(char)
                continue
            for symbol in morse:
                morse_array.append(symbol)
            is_last_letter = (letter_index == len(word) - 1)
            is_last_word   = (word_index   == len(words) - 1)
            if is_last_letter and is_last_word:
                pass
            elif is_last_letter:
                morse_array.append("||")
            else:
                morse_array.append("|")

    parts = []
    current = []
    for symbol in morse_array:
        if symbol in ("|", "||"):
            if current:
                parts.append("".join(current))
                current = []
            parts.append(symbol)
        else:
            current.append(symbol)
    if current:
        parts.append("".join(current))
    sequence_string = " ".join(parts)

    return morse_array, sequence_string, skipped

# =============================================================================
# Main Function, enter EMG Raw Data CSV file
# =============================================================================

if __name__ == "__main__":
    while True:
        print("\n" + "=" * 50)
        print("SELECT INPUT MODE")
        print("=" * 50)
        print("  1 - Run Test 1 (hi)")
        print("  2 - Run Test 2 (how are you)")
        print("  3 - Custom Text Input")
        print("  q - Quit")
        print("=" * 50)
        choice = input("\nEnter choice (1, 2, 3, or q): ").strip()

        if choice == "1":
            print_config()
            detector, avg_mav = process_emg_file("test_hi_morse_with_noise.csv")
            print("\n--- RESULTS ---")
            print(f"Array sequence: {detector.get_sequence()}\n")
            print(f"Detected sequence: {detector.get_output_string()}\n")
            print("Morse Code Translation:", translate_morse(detector.get_sequence()), '\n')

        elif choice == "2":
            print_config()
            detector, avg_mav = process_emg_file("test_how_are_you_morse.csv")
            print("\n--- RESULTS ---")
            print(f"Array sequence: {detector.get_sequence()}\n")
            print(f"Detected sequence: {detector.get_output_string()}\n")
            print("Morse Code Translation:", translate_morse(detector.get_sequence()), '\n')

        elif choice == "3":
            user_input = input("\nEnter text to convert: ").strip()
            if not user_input:
                print("No input provided.")
            else:
                morse_array, sequence_string, skipped = text_to_morse_array(user_input)
                print("\n--- RESULTS ---")
                print(f"Custom Input Detected: {user_input}")
                if skipped:
                    print(f"  [!] Skipped unsupported characters: {skipped}")
                print(f"\nMorse Code Array sequence:\n{morse_array}\n")
                print(f"Morse Code Detected sequence:\n{sequence_string}\n")

        elif choice == "q":
            print("Exiting.")
            break

        else:
            print("Invalid choice. Please enter 1, 2, 3, or q.")