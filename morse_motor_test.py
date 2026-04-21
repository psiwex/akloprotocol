import time
import lgpio

GPIO_PIN = 27

DOT_DURATION = 0.8  
DASH_DURATION = 2.0   
INTERVAL = 3.0         


# ==================== GPIO Reset ====================

def reset_gpio():
    try:
        h = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(h, GPIO_PIN)
        lgpio.gpio_write(h, GPIO_PIN, 0)
        time.sleep(0.1)
        lgpio.gpiochip_close(h)
        print("✓ GPIO reset complete\n")
    except:
        pass


# ==================== Motor Control ====================

def vibrate_motor(pattern):
    """
    pattern: list like ['dot', 'dash', 'dot']
    """

    h = None
#     MIN_ON = 0.2
#     duration = 0
#     on_time = max(duration, MIN_ON)
    

    try:
        h = lgpio.gpiochip_open(4)
        lgpio.gpio_claim_output(h, GPIO_PIN)

        lgpio.gpio_write(h, GPIO_PIN, 0)
        time.sleep(0.1)

        print("Starting motor test...\n")

        for i, symbol in enumerate(pattern):

            if symbol == 'dot':
                duration = DOT_DURATION
            elif symbol == 'dash':
                duration = DASH_DURATION
            else:
                print(f"Unknown symbol: {symbol}")
                continue

            print(f"[{i}] {symbol.upper()} → Motor ON")
            lgpio.gpio_write(h, GPIO_PIN, 1)
            time.sleep(duration)

            lgpio.gpio_write(h, GPIO_PIN, 0)
            print(f"[{i}] Motor OFF")

            time.sleep(INTERVAL)

        print("\n✓ Motor test completed")

    except Exception as e:
        print(f"GPIO Error: {e}")

    finally:
        if h is not None:
            lgpio.gpio_write(h, GPIO_PIN, 0)
            time.sleep(0.05)
            lgpio.gpiochip_close(h)
            print("✓ GPIO cleaned up")


# ==================== Main ====================

if __name__ == "__main__":

    reset_gpio()

    test_pattern = ['dot', 'dash','dot','dot','dash']

    vibrate_motor(test_pattern)