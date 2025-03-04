#!/usr/bin/env python3
import RPi.GPIO as GPIO
import time
import os

# Print Start of test
print("ReSpeaker 2-Mic Pi Button - Shutdown Script")

# User button pin
BUTTON = 17

# User button setup
GPIO.setmode(GPIO.BCM)
GPIO.setup(BUTTON, GPIO.IN)

# Save previous state
previousState = GPIO.input(BUTTON)

try:
    # Unending Loop (main)
    while True:
        # Get button state
        currentState = GPIO.input(BUTTON)

        # Check if any difference
        if currentState != previousState:
            # Store current state as previous
            previousState = currentState

            # Check the current state of the button
            if currentState:
                print("Button is not clicked")
            else:
                print("Button is clicked - Shutting down!")
                os.system("sudo shutdown -h now")  # Shutdown command
                break  # Exit loop after triggering shutdown

        time.sleep(0.1)  # Poll every 100ms

except KeyboardInterrupt:
    print("Script interrupted. Cleaning up GPIO...")
    GPIO.cleanup()
