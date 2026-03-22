#!/usr/bin/env python3
"""Quick test script for Philips Hue bridge connection and light control.

Usage:
  1. Press the button on your Hue bridge
  2. Run: python test_hue.py
"""

import time
from phue import Bridge

BRIDGE_IP = "10.0.0.2"

print(f"Connecting to Hue bridge at {BRIDGE_IP}...")
print("(If this is the first time, press the bridge button NOW)")
bridge = Bridge(BRIDGE_IP)
bridge.connect()
print("Connected!\n")

# List all lights
lights = bridge.lights
print(f"Found {len(lights)} light(s):")
for light in lights:
    print(f"  ID: {light.light_id}  Name: {light.name}  On: {light.on}")

if not lights:
    print("No lights found. Make sure bulbs are paired with the bridge.")
    exit(1)

light_ids = [l.light_id for l in lights]

def set_all(cmd):
    for lid in light_ids:
        bridge.set_light(lid, cmd)

print("\n--- Running light demo ---\n")

# Turn on
print("1. Turning lights ON (white)")
set_all({"on": True, "bri": 254, "sat": 0, "hue": 0, "transitiontime": 5})
time.sleep(2)

# Red
print("2. Red")
set_all({"on": True, "hue": 0, "sat": 254, "bri": 254, "transitiontime": 5})
time.sleep(2)

# Green
print("3. Green")
set_all({"on": True, "hue": 21845, "sat": 254, "bri": 254, "transitiontime": 5})
time.sleep(2)

# Blue
print("4. Blue")
set_all({"on": True, "hue": 43690, "sat": 254, "bri": 254, "transitiontime": 5})
time.sleep(2)

# Warm yellow (excited)
print("5. Warm yellow (excited emotion)")
set_all({"on": True, "hue": 7600, "sat": 254, "bri": 254, "transitiontime": 5})
time.sleep(2)

# Flash effect
print("6. Flash effect")
for _ in range(3):
    set_all({"bri": 254, "transitiontime": 1})
    time.sleep(0.2)
    set_all({"bri": 50, "transitiontime": 1})
    time.sleep(0.2)
time.sleep(1)

# Breathing effect
print("7. Breathing effect (5 seconds)")
set_all({"on": True, "hue": 8000, "sat": 140, "bri": 140, "transitiontime": 10})
time.sleep(1)
for _ in range(2):
    set_all({"bri": 180, "transitiontime": 15})
    time.sleep(1.5)
    set_all({"bri": 60, "transitiontime": 15})
    time.sleep(1.5)

# Back to calm white
print("8. Calm white")
set_all({"on": True, "hue": 8000, "sat": 80, "bri": 180, "transitiontime": 10})
time.sleep(2)

print("\nDemo complete! Lights left on. Run with --off to turn off.")

import sys
if "--off" in sys.argv:
    print("Turning lights off...")
    set_all({"on": False, "transitiontime": 10})
