import pandas as pd
import numpy as np
import time
import lgpio
import matplotlib.pyplot as plt
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# ---------- EMG / Morse Detection ----------
THRESHOLD = 0.00005022

SAMPLING_RATE = 1000
WINDOW_SIZE = 200
OVERLAP = 0.5

STEP_SIZE = int(WINDOW_SIZE * (1 - OVERLAP))
MAV_INTERVAL = STEP_SIZE / SAMPLING_RATE

MIN_ACTIVATION = 0.5
DASH_THRESHOLD = 2.0

MIN_RELEASE = 0.5
LETTER_GAP = 3.0
WORD_GAP = 6.0

MIN_ACTIVATION_SAMPLES = int(MIN_ACTIVATION / MAV_INTERVAL)
DASH_THRESHOLD_SAMPLES = int(DASH_THRESHOLD / MAV_INTERVAL)
MIN_RELEASE_SAMPLES = int(MIN_RELEASE / MAV_INTERVAL)
LETTER_GAP_SAMPLES = int(LETTER_GAP / MAV_INTERVAL)
WORD_GAP_SAMPLES = int(WORD_GAP / MAV_INTERVAL)

CHANNEL_PREFIX = "channel"

# ---------- GPIO / Haptic ----------
GPIO_PIN = 17

DOT_DURATION = 0.30
DASH_DURATION = 0.80

SYMBOL_GAP = 1.00
LETTER_INTERVAL = 3.00   # "|" gap
WORD_INTERVAL = 6.00     # "||" gap

SETTLE_TIME = 0.10


# =============================================================================
# GPIO RESET
# =============================================================================

def reset_gpio():
    """Force reset GPIO17 to ensure clean state"""
    try:
        h = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(h, GPIO_PIN)
        lgpio.gpio_write(h, GPIO_PIN, 0)
        time.sleep(0.1)
        lgpio.gpiochip_close(h)
        print("✓ GPIO reset complete\n")
    except:
        pass


# =============================================================================
# MAV CALCULATION
# =============================================================================

def calculate_mav(channel_data):
    channel_mavs = []

    for ch_name, signal in channel_data.items():
        mav_values = []
        start = 0

        while start + WINDOW_SIZE <= len(signal):
            window = signal[start:start + WINDOW_SIZE]
            mav = np.mean(np.abs(window))
            mav_values.append(mav)
            start += STEP_SIZE

        channel_mavs.append(mav_values)

    all_mavs = np.array(channel_mavs)
    averaged_mav = np.mean(all_mavs, axis=0)

    return averaged_mav


# =============================================================================
# MORSE DETECTOR
# =============================================================================

class MorseDetector:
    def __init__(self):
        self.state = "IDLE"
        self.active_count = 0
        self.gap_count = 0
        self.current_letter = []
        self.sequence = []
        self.output = []

    def process_mav(self, mav_value):
        is_active = mav_value > THRESHOLD
        event = None

        if self.state == "IDLE":
            if is_active:
                self.state = "ACTIVE"
                self.active_count = 1

        elif self.state == "ACTIVE":
            if is_active:
                self.active_count += 1
            else:
                event = self._classify_activation()
                self.state = "GAP"
                self.gap_count = 1
                self.active_count = 0

        elif self.state == "GAP":
            if is_active:
                self.state = "ACTIVE"
                self.active_count = 1
                self.gap_count = 0
            else:
                self.gap_count += 1

                if self.gap_count == LETTER_GAP_SAMPLES:
                    event = self._finalize_letter()
                elif self.gap_count == WORD_GAP_SAMPLES:
                    event = self._finalize_word()

        return event

    def _classify_activation(self):
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
        if self.current_letter:
            letter = "".join(self.current_letter)
            self.output.append(letter)
            self.sequence.append("|")
            self.current_letter = []
            return f"LETTER COMPLETE: {letter}"
        return None

    def _finalize_word(self):
        if self.output and self.output[-1] != " ":
            self.output.append(" ")
            if self.sequence and self.sequence[-1] == "|":
                self.sequence[-1] = "||"
            return "WORD BOUNDARY"
        return None

    def finalize(self):
        if self.current_letter:
            letter = "".join(self.current_letter)
            self.output.append(letter)
            self.current_letter = []

    def get_output(self):
        return self.output

    def get_output_string(self):
        return " | ".join(self.output)

    def get_sequence(self):
        return self.sequence

    def get_sequence_string(self):
        return "".join(self.sequence)


# =============================================================================
# MORSE TRANSLATION
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
        "-.--":"y", "--..":"z", ".----":"1", "..---":"2",
        "...--":"3", "....-":"4", ".....":"5", "-....":"6",
        "--...":"7", "---..":"8", "----.":"9", "-----":"0"
    }

    while i < len(morse_array):
        if morse_array[i] == "||":
            translation += dictionary.get(current_letter, "?")
            translation += " "
            current_letter = ""
        elif morse_array[i] == "|":
            translation += dictionary.get(current_letter, "?")
            current_letter = ""
        else:
            current_letter += morse_array[i]
        i += 1

    if current_letter:
        translation += dictionary.get(current_letter, "?")

    return translation


# =============================================================================
# FILE PROCESSING
# =============================================================================

def process_emg_file(filepath):
    df = pd.read_csv(filepath)

    channel_columns = [col for col in df.columns if col.startswith(CHANNEL_PREFIX)]
    if len(channel_columns) == 0:
        available = list(df.columns)
        raise ValueError(
            f"No columns found starting with '{CHANNEL_PREFIX}'. Available columns: {available}"
        )

    channel_columns = sorted(
        channel_columns,
        key=lambda x: int(x.replace(CHANNEL_PREFIX, "") or 0)
    )

    print(f"Found {len(channel_columns)} channels: {channel_columns}")

    channel_data = {col: df[col].values for col in channel_columns}

    num_samples = len(df)
    print(f"Loaded {num_samples} samples ({num_samples / SAMPLING_RATE:.2f} seconds)")

    averaged_mav = calculate_mav(channel_data)
    print(f"Calculated {len(averaged_mav)} averaged MAV values")

    detector = MorseDetector()

    print("\n--- Processing ---")
    for i, mav in enumerate(averaged_mav):
        event = detector.process_mav(mav)
        if event:
            time_sec = i * MAV_INTERVAL
            print(f"[{time_sec:6.2f}s] {event}")

    detector.finalize()
    return detector, averaged_mav


# =============================================================================
# EXPECTED HAPTIC TIMELINE GRAPH
# =============================================================================

def build_expected_timeline(sequence):
    """
    Build ideal binary timeline:
    vibration ON  -> 1
    vibration OFF -> 0
    """
    times = [0.0]
    states = [0]
    current_time = 0.0

    for symbol in sequence:
        if symbol == ".":
            on_start = current_time
            on_end = current_time + DOT_DURATION

            times.extend([on_start, on_end])
            states.extend([1, 1])

            current_time = on_end

            off_end = current_time + SETTLE_TIME + SYMBOL_GAP
            times.extend([current_time, off_end])
            states.extend([0, 0])

            current_time = off_end

        elif symbol == "-":
            on_start = current_time
            on_end = current_time + DASH_DURATION

            times.extend([on_start, on_end])
            states.extend([1, 1])

            current_time = on_end

            off_end = current_time + SETTLE_TIME + SYMBOL_GAP
            times.extend([current_time, off_end])
            states.extend([0, 0])

            current_time = off_end

        elif symbol == "|":
            off_end = current_time + LETTER_INTERVAL
            times.extend([current_time, off_end])
            states.extend([0, 0])

            current_time = off_end

        elif symbol == "||":
            off_end = current_time + WORD_INTERVAL
            times.extend([current_time, off_end])
            states.extend([0, 0])

            current_time = off_end

    return times, states, current_time


def save_expected_timeline_graph(sequence, csv_name):
    """
    Save expected haptic binary timeline graph to current folder.
    """
    times, states, total_time = build_expected_timeline(sequence)

    base_name = Path(csv_name).stem
    output_png = f"{base_name}_haptic_expected_timeline.png"

    plt.figure(figsize=(12, 4))
    plt.step(times, states, where='post')
    plt.ylim(-0.2, 1.2)
    plt.xlim(0, max(total_time, 0.1))
    plt.yticks([0, 1])
    plt.xlabel("Time (s)")
    plt.ylabel("Motor State")
    plt.title("Expected Haptic Output Timeline")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300)
    plt.close()

    return output_png


# =============================================================================
# HAPTIC PLAYBACK
# =============================================================================

def play_haptic_from_sequence(sequence):
    """
    Keep original sequence output unchanged.
    Only convert internally for vibration:
      .  -> dot
      -  -> dash
      |  -> letter gap
      || -> word gap
    """
    h = None

    try:
        h = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(h, GPIO_PIN)
        lgpio.gpio_write(h, GPIO_PIN, 0)
        time.sleep(0.1)

        print("\n" + "=" * 60)
        print("Starting haptic playback")
        print("=" * 60 + "\n")

        for idx, symbol in enumerate(sequence):
            if symbol == ".":
                lgpio.gpio_write(h, GPIO_PIN, 1)
                print(f"[{idx}] dot", end="", flush=True)
                time.sleep(DOT_DURATION)
                lgpio.gpio_write(h, GPIO_PIN, 0)
                time.sleep(SETTLE_TIME)
                time.sleep(SYMBOL_GAP)
                print(" ✓")

            elif symbol == "-":
                lgpio.gpio_write(h, GPIO_PIN, 1)
                print(f"[{idx}] dash", end="", flush=True)
                time.sleep(DASH_DURATION)
                lgpio.gpio_write(h, GPIO_PIN, 0)
                time.sleep(SETTLE_TIME)
                time.sleep(SYMBOL_GAP)
                print(" ✓")

            elif symbol == "|":
                lgpio.gpio_write(h, GPIO_PIN, 0)
                print(f"[{idx}] letter gap ({LETTER_INTERVAL:.1f}s)", end="", flush=True)
                time.sleep(LETTER_INTERVAL)
                print(" ✓")

            elif symbol == "||":
                lgpio.gpio_write(h, GPIO_PIN, 0)
                print(f"[{idx}] word gap ({WORD_INTERVAL:.1f}s)", end="", flush=True)
                time.sleep(WORD_INTERVAL)
                print(" ✓")

        print("\n" + "=" * 60)
        print("✓ Haptic playback completed")
        print("=" * 60)

    except Exception as e:
        print(f"\n✗ GPIO Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        if h is not None:
            try:
                lgpio.gpio_write(h, GPIO_PIN, 0)
                time.sleep(0.05)
                lgpio.gpiochip_close(h)
                print("\n✓ GPIO cleaned up properly")
            except Exception as e:
                print(f"\nWarning during cleanup: {e}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("Initializing...")
    reset_gpio()

    csv_files = list(Path('.').glob('*.csv'))
    if not csv_files:
        print("No CSV files found")
        return

    print("Available files:")
    for i, f in enumerate(csv_files, 1):
        print(f"  {i}. {f.name}")

    choice = int(input(f"\nSelect (1-{len(csv_files)}): ")) - 1
    csv_file = csv_files[choice]

    print(f"\nProcessing file: {csv_file.name}\n")

    detector, averaged_mav = process_emg_file(csv_file)

    output_array = detector.get_output()
    output_string = detector.get_output_string()
    sequence = detector.get_sequence()
    sequence_string = detector.get_sequence_string()
    translation = translate_morse(sequence)

    print("\n--- Results ---")
    print(f"Output array:      {output_array}")
    print(f"Output string:     {output_string}")
    print(f"Sequence array:    {sequence}")
    print(f"Sequence string:   {sequence_string}")
    print(f"Translation:       {translation}")

    graph_file = save_expected_timeline_graph(sequence, csv_file.name)
    print(f"Expected timeline graph saved: {graph_file}")

    play_haptic_from_sequence(sequence)


if __name__ == "__main__":
    main()