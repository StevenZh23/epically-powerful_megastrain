"""epically-powerful module for managing IMUs.

This module contains functions for calibrating the 
accelerometer, gyroscope, and magnetometer of the Megastrain5000 IMUs. 
Methodology for these steps is adapted from the MPU9250 calibration steps

"""

import os
import sys
import time
import platform
import argparse
import json
from typing import Dict
import numpy as np
from numpy.typing import NDArray
from scipy.optimize import curve_fit
import smbus2 as smbus # I2C bus library on Raspberry Pi and NVIDIA Jetson Orin Nano
from epicallypowerful.toolbox import LoopTimer
from epicallypowerful.sensing.megastrain5000.megastrain_imu import MEGASTRAIN5000IMUs


# Get default I2C bus depending on which device is currently being used
machine_name = platform.uname().release.lower()

if "tegra" in machine_name:
    DEFAULT_I2C_BUS = 7
elif "rpi" in machine_name or "bcm" in machine_name or "raspi" in machine_name:
    DEFAULT_I2C_BUS = 1
else:
    DEFAULT_I2C_BUS = 0


# Specify constants
G = 9.80665 # [m*s^-2]


def get_linear_output(
    data: NDArray[np.float64],
    m: float,
    b: float,
) -> NDArray[np.float64]:
    return (data * m) + b


def remove_outliers(
    data: NDArray[np.float64],
    std_scaler: float,
) -> NDArray[np.float64]:
    inlier_data = data
    data_diff = np.append(
        np.zeros(1),
        np.diff(data),
    )

    outlier_idxs = np.abs(data_diff) > np.abs(np.mean(data_diff)) + (std_scaler * np.std(data_diff))
    inlier_data[outlier_idxs] = np.nan

    return inlier_data


def calibrate_accelerometer(
    imu_handler: MEGASTRAIN5000IMUs,
    loop_timer: LoopTimer,
    time_to_calibrate: float=2.5,
    verbose: bool=False,
) -> list[list[float]]:
    acc_offset_coeffs = [[], [], []]
    directions_to_calibrate = ['face-up', 'face-down', 'perpendicular to gravity']
    axes_to_calibrate = ['x', 'y', 'z']

    for a, axis_to_cal in enumerate(axes_to_calibrate):
        axis_offsets = [[], [], []]

        for d, dir_to_cal in enumerate(directions_to_calibrate):
            input(f"Hold IMU accelerometer {axis_to_cal} axis {dir_to_cal}, then press [ENTER]...")
            acc_data = []
            t_start = time.perf_counter()

            while time.perf_counter() - t_start <= time_to_calibrate:
                if loop_timer.continue_loop():
                    imu_data = imu_handler.get_data(imu_id=0)
                    acc_data.append([imu_data.acc_x, imu_data.acc_y, imu_data.acc_z])

                    # Show progress of calibration
                    print(f"Calibrating: {round(((time.perf_counter() - t_start)/time_to_calibrate) * 100)}%\r", end="")


            axis_offsets[d] = np.array(acc_data)[:, a]

        optimized_params, _ = curve_fit(
            f=get_linear_output,
            xdata=np.array(axis_offsets).flatten(),
            ydata=np.concatenate([
                np.ones(np.shape(axis_offsets[0])) * G,  # +9.81 m*s^-2
                -np.ones(np.shape(axis_offsets[1])) * G, # -9.81 m*s^-2
                np.zeros(np.shape(axis_offsets[2])) * G, # 0 g
            ]),
            maxfev=10000,
        )

        acc_offset_coeffs[a] = optimized_params.tolist()


    if verbose:
        # print(f"acc_offset_coeffs: {[acc:0.2f for acc in acc_offset_coeffs[0]]}")
        print(f"Calibration complete! acc_offset_coeffs: ({acc_offset_coeffs[0]}, {acc_offset_coeffs[1]}, {acc_offset_coeffs[2]})")

    return acc_offset_coeffs


def calibrate_gyroscope(
    imu_handler: MEGASTRAIN5000IMUs,
    loop_timer: LoopTimer,
    time_to_calibrate: float=2.5,
    verbose: bool=False,
) -> tuple[float]:
    input("Keep IMU still, then press [ENTER] to calibrate gyroscope...")
    gyro_data = []
    t_start = time.perf_counter()

    while time.perf_counter() - t_start <= time_to_calibrate:
        if loop_timer.continue_loop():
            imu_data = imu_handler.get_data(imu_id=0)
            gyro_data.append([imu_data.gyro_x, imu_data.gyro_y, imu_data.gyro_z])

            # Show progress of calibration
            print(f"Calibrating: {round(((time.perf_counter() - t_start)/time_to_calibrate) * 100)}%\r", end="")

    if verbose:
        print(f"gyro_data size for calibration: {len(gyro_data)} samples")

    gyro_coeffs = np.mean(np.array(gyro_data), 0).tolist()

    if verbose:
        print(f"Calibration complete! gyro_coeffs: ({gyro_coeffs[0]:0.2f}, {gyro_coeffs[1]:0.2f}, {gyro_coeffs[2]:0.2f})")

    return gyro_coeffs


def split_strings(arg):
    return arg.split(",")


# Create argument parser
parser = argparse.ArgumentParser(
    description="Calibrate MEGASTRAIN5000 IMU."
)

parser.add_argument(
    "--components",
    type=split_strings,
    default="acc,gyro",
    help="Which channels to calibrate on the sensor, comma-separated without spaces (e.g., --components acc,gyro)",
)

parser.add_argument(
    "--i2c-bus",
    type=int,
    default=DEFAULT_I2C_BUS,
    help="Which I2C bus the sensor is on (e.g., --i2c-bus 7). Defaults to 7 for NVIDIA Jetson Orin Nano or 1 for Raspberry Pi",
)

parser.add_argument(
    "--channel",
    type=int,
    default=-1,
    help="(If using multiplexer) channel sensor is on (e.g., --channel 1). Defaults to -1 (no multiplexer)",
)

parser.add_argument(
    "--address",
    type=int,
    default=68,
    help="I2C address of MEGASTRAIN5000 sensor (e.g., --address 68). Defaults to 68",
)

parser.add_argument(
    "--rate",
    type=float,
    default=400,
    help="Frequency of MEGASTRAIN5000 sensor [Hz] (e.g., --rate 400). Defaults to 400 Hz",
)


if __name__ == "__main__":
    # Pull in command-line arguments for which IMU to calibrate and how to do so
    args = parser.parse_args()
    bus = args.i2c_bus
    channel = args.channel
    address = int('0x'+str(args.address), 0) # Convert multi-digit address into hex, then int
    components = args.components
    print(f"components: {components}")
    print(f"Initializing MEGASTRAIN5000 IMU at I2C bus {bus} on channel {channel} with address {address}")

    # Set up dictionary for IMU currently being calibrated
    # TODO: decide whether to add functionality for calibrating multiple IMUs in sequence
    imu_id = {
        0:
            {
                'bus': bus,
                'channel': channel,
                'address': address,
                'acc': [],
                'gyro': [],
            },
    }

    # Look for existing calibration for current IMU in JSON file
    calibration_filename = f"megastrain5000_calibrations.json"
    
    if os.path.isfile(calibration_filename):
        with open(calibration_filename, "r") as f:
            calibration_dict = json.load(f)
    else:
        calibration_dict = {f"{bus}_{channel}_{address}": {}}

        for component in imu_id[0].keys():
            calibration_dict[f"{bus}_{channel}_{address}"][component] = imu_id[0][component]
    
    # TROUBLESHOOTING
    # print(f"calibration_dict: \n{calibration_dict}")

    # Create instance of MEGASTRAIN5000 IMUs object to initialize connected IMU
    megastrain5000_imus = MEGASTRAIN5000IMUs(
        imu_ids=imu_id,
        components=args.components,
        calibration_path='', # When generating a new calibration, don't rely on old one
        verbose=True,
    )

    for component in args.components:
        calibrate_component = False
        imu_calibration_exists = False
        print(f"Checking calibration component {component}...")

        # Get user confirmation on whether to overwrite existing calibration (if one already exists)
        for idx in calibration_dict.keys():
            # Check all IMUs in dict to see whether they are connected in 
            # the configuration to be the IMU currently being calibrated
            if calibration_dict[idx]['bus'] == bus and calibration_dict[idx]['channel'] == channel and calibration_dict[idx]['address'] == address:

                if len(calibration_dict[idx][component]) > 0:
                    imu_calibration_exists = True
                    user_input = input(f"Overwrite existing {component} calibration for IMU (bus: {bus}, channel: {channel}, address: {address})? Enter [y/n], then press [ENTER] to continue: ")

                    if user_input == 'y':
                        calibrate_component = True
                    else:
                        print(f"Skipping {component} calibration for IMU (bus: {bus}, channel: {channel}, address: {address})...")
                else:
                    calibrate_component = True
        
        if not imu_calibration_exists:
            calibrate_component = True

        if not calibrate_component:
            continue

        if component == "acc":
            # TROUBLESHOOTING
            print("Getting acceleration calibration!")

            imu_id[0]['acc'] = calibrate_accelerometer(
                imu_handler=megastrain5000_imus,
                loop_timer=LoopTimer(operating_rate=args.rate, verbose=False),
                time_to_calibrate=2.5,
                verbose=True,
            )

        elif component == "gyro":
            # TROUBLESHOOTING
            print("Getting gyroscope calibration!")
            
            imu_id[0]['gyro'] = calibrate_gyroscope(
                imu_handler=megastrain5000_imus,
                loop_timer=LoopTimer(operating_rate=args.rate, verbose=True),
                time_to_calibrate=2.5,
                verbose=True,
            )
        else:
            print(f"Component {component} not of type `acc`, `gyro`, or `mag`. Skipping...")

        # Iterate through calibration dict, updating calibration components where necessary
        for idx in calibration_dict.keys():
            # Index IMUs in calibration dict by bus, channel, and address configuration
            if calibration_dict[idx]['bus'] == bus and calibration_dict[idx]['channel'] == channel and calibration_dict[idx]['address'] == address:
                calibration_dict[idx][component] = imu_id[0][component]

    # Save calibration dictionary to same JSON file
    calibration_dict_str = json.dumps(calibration_dict, indent=4)

    with open(calibration_filename, "w") as f:
        f.write(calibration_dict_str)
