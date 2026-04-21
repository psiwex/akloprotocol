import lgpio
import time

# GPIO
GPIO_PIN = 17
DOT_DURATION    = 0.80
DASH_DURATION   = 2.00
SYMBOL_GAP      = 1.00
LETTER_INTERVAL = 3.00
WORD_INTERVAL   = 6.00
SETTLE_TIME     = 0.10

MORSE_DICT = {
    "a": ".-",   "b": "-...", "c": "-.-.", "d": "-..",
    "e": ".",    "f": "..-.", "g": "--.",  "h": "....",
    "i": "..",   "j": ".---", "k": "-.-", "l": ".-..",
    "m": "--",   "n": "-.",   "o": "---", "p": ".--.",
    "q": "--.-", "r": ".-.",  "s": "...", "t": "-",
    "u": "..-",  "v": "...-", "w": ".--", "x": "-..-",
    "y": "-.--", "z": "--.."
}

def text_to_sequence(text):
    sequence = []
    words = text.lower().strip().split()
    for w_idx, word in enumerate(words):
        for l_idx, letter in enumerate(word):
            for symbol in MORSE_DICT.get(letter, ""):
                sequence.append(symbol)
            if l_idx < len(word) - 1:
                sequence.append("|")
        if w_idx < len(words) - 1:
            sequence.append("||")
    return sequence

def play_haptic(sequence):
    h = None
    try:
        h = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(h, GPIO_PIN)
        lgpio.gpio_write(h, GPIO_PIN, 0)
        time.sleep(0.5)

        for idx, symbol in enumerate(sequence):
            if symbol == ".":
                lgpio.gpio_write(h, GPIO_PIN, 1)
                print(f"[{idx}] dot", end="", flush=True)
                time.sleep(DOT_DURATION)
                lgpio.gpio_write(h, GPIO_PIN, 0)
                time.sleep(SETTLE_TIME + SYMBOL_GAP)
                print(" ✓")

            elif symbol == "-":
                lgpio.gpio_write(h, GPIO_PIN, 1)
                print(f"[{idx}] dash", end="", flush=True)
                time.sleep(DASH_DURATION)
                lgpio.gpio_write(h, GPIO_PIN, 0)
                time.sleep(SETTLE_TIME + SYMBOL_GAP)
                print(" ✓")

            elif symbol == "|":
                print(f"[{idx}] letter gap ({LETTER_INTERVAL:.1f}s)", end="", flush=True)
                time.sleep(LETTER_INTERVAL)
                print(" ✓")

            elif symbol == "||":
                print(f"[{idx}] word gap ({WORD_INTERVAL:.1f}s)", end="", flush=True)
                time.sleep(WORD_INTERVAL)
                print(" ✓")

    finally:
        if h is not None:
            lgpio.gpio_write(h, GPIO_PIN, 0)
            lgpio.gpiochip_close(h)

if __name__ == "__main__":
    time.sleep(5)
    user_input = input("Enter text to convert: ")
    sequence = text_to_sequence(user_input)
    print(f"Input: {user_input}")
    print(f"Sequence: {sequence}")
    play_haptic(sequence)