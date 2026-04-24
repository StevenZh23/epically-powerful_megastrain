"""epically-powerful module for managing IMUs.

This module contains the classes and commands for initializing
and reading from MEGASTRAIN5000 IMUs.
"""

import os
import sys
import time
import json
from typing import Dict
import smbus2 as smbus # I2C bus library on Raspberry Pi and NVIDIA Jetson Orin Nano
from epicallypowerful.toolbox import LoopTimer
from epicallypowerful.sensing.imu_data import IMUData
from epicallypowerful.sensing.imu_abc import IMU

# Unit conversions
PI = 3.1415926535897932384
GRAV_ACC = 9.81 # [m*s^-2]
DEG2RAD = PI/180
RAD2DEG = 180/PI

# Set ICM42607 (accelerometer) registers
WHO_AM_I_ICM = 0x75
ICM42607_ADDR = 0x68
ICM42607_ADDR_AD0_HIGH = 0x69
PWR_MGMT_0   = 0x1F
GYRO_CONFIG_0  = 0x20
GYRO_CONFIG_1  = 0x23
ACCEL_CONFIG_0 = 0x21
ACCEL_CONFIG_1 = 0x24

INT_CONFIG_ICM = 0x06

ACCEL_XOUT_H = 0x0B
ACCEL_YOUT_H = 0x0D
ACCEL_ZOUT_H = 0x0F
GYRO_XOUT_H  = 0x11
GYRO_YOUT_H  = 0x13
GYRO_ZOUT_H  = 0x15

# Set constants
ACC_RANGE_2G    = 0 # Set MEGASTRAIN5000 accelerometer resolution to +/- 2 g's
ACC_RANGE_4G    = 1 # Set MEGASTRAIN5000 accelerometer resolution to +/- 4 g's
ACC_RANGE_8G    = 2 # Set MEGASTRAIN5000 accelerometer resolution to +/- 8 g's
ACC_RANGE_16G   = 3 # Set MEGASTRAIN5000 accelerometer resolution to +/- 16 g's
GYRO_RANGE_250_DEG_PER_S  = 0 # Set MEGASTRAIN5000 gyroscope resolution to +/- 250.0 deg/s
GYRO_RANGE_500_DEG_PER_S  = 1 # Set MEGASTRAIN5000 gyroscope resolution to +/- 500.0 deg/s
GYRO_RANGE_1000_DEG_PER_S = 2 # Set MEGASTRAIN5000 gyroscope resolution to +/- 1000.0 deg/s
GYRO_RANGE_2000_DEG_PER_S = 3 # Set MEGASTRAIN5000 gyroscope resolution to +/- 2000.0 deg/s
SLEEP_TIME   = 0.1 # [s]


class MEGASTRAIN5000IMUs(IMU):
    """Class for interfacing with the MEGASTRAIN5000 IMU using I2C communication, leveraging the TCA9548A multiplexer for communicating with multiple units at the same time.

    This class draws from the following resources:
        - MPU9250 calibration: https://github.com/makerportal/mpu92-calibration
        - TCA9548a multiplexer to connect multiple I2C devices with the same address: https://wolles-elektronikkiste.de/en/tca9548a-i2c-multiplexer
        - TDK InvenSense MPU9250 datasheet: https://invensense.tdk.com/wp-content/uploads/2015/02/PS-MPU-9250A-01-v1.1.pdf
        - PCA9548A datasheet: https://www.ti.com/lit/ds/symlink/pca9548a.pdf
        - Chase Sun's work with the MEGASTRAIN5000 and ICM42607 IMUs

    Many helper functions are included in the :py:class:`IMUData` class to assist with getting data conveniently. Please see that documentation for all options.

    Example:
        .. code-block:: python

            from epicallypowerful.sensing import MEGASTRAIN5000IMUs

            ### Instantiation ---
            imu_ids = {
                0: {
                    'bus': 1,
                    'channel': -1, # -1 --> no multiplexer, otherwise --> multiplexer channel
                    'address': 0x68,
                },
                1: {
                    'bus': 1,
                    'channel': -1,
                    'address': 0x69,
                },
            }

            imus = MEGASTRAIN5000IMUs(
                imu_ids=imu_ids,
                components=['acc', 'gyro'],
            )

            ### Stream data ---
            print(imus.get_data(imu_id=0).acc_x)
            print(imus.get_data(imu_id=1).acc_x)

    Args:
        imu_ids (dict): dictionary of each IMU and the I2C bus number, multiplexer channel (if used), and I2C address needed to access it.
        components (list of strings): list of MEGASTRAIN5000 sensing components to get. Could include `acc` or `gyro`.
        acc_range_selector (int): index for range of accelerations to collect. Default: 2 (+/- 8 g), but can be:
            0: +/- 2.0 g's
            1: +/- 4.0 g's
            2: +/- 8.0 g's
            3: +/- 16.0 g's
        gyro_range_selector (int): index for range of angular velocities to collect. Default: 1 (+/- 500.0 deg/s), but can be:
            0: +/- 250.0 deg/s
            1: +/- 500.0 deg/s
            2: +/- 1000.0 deg/s
            3: +/- 2000.0 deg/s
        calibration_path (str): path to JSON file with calibration values for IMUs to be connected. NOTE: this file indexes IMUs by which bus, multiplexer channel (if used), and I2C address they are connected to. Be careful not to use the calibration for one IMU connected in this way on another unit by mistake.
        verbose (bool): whether to print verbose output from IMU operation. Default: False.
    """

    def __init__(
        self,
        imu_ids: dict[int, dict[str, int]],
        components=['acc','gyro'],
        acc_range_selector=ACC_RANGE_8G,
        gyro_range_selector=GYRO_RANGE_500_DEG_PER_S,
        calibration_path='',
        verbose: bool=False,
    ) -> None:
        if imu_ids is None:
            raise Exception('`imu_ids` must contain at least one IMU index.')
        elif not isinstance(imu_ids,dict):
            raise Exception ('`imu_ids` must be in the form of dict(int, dict(int bus_id, int channel, hex imu_id).')

        # Initialize all IMU-specific class attributes
        self.imu_ids = imu_ids
        self.components = components
        self.acc_range_selector = acc_range_selector
        self.gyro_range_selector = gyro_range_selector
        self.verbose = verbose
        self.bus = {}
        self.calibration_dict = {}

        # Look for existing calibrations for IMUs
        if len(calibration_path) > 0:
            if self.verbose:
                print(f"Looking for calibration at: {calibration_path}")
            
            if os.path.isfile(calibration_path):                
                with open(calibration_path, "r") as f:
                    self.calibration_dict = json.load(f)

                if self.verbose:
                    print(f"Found calibration file!")
            else:        
                if self.verbose:
                    print("No calibration file found. Proceeding with raw values...")
        else:
            if self.verbose:
                print(f"No calibration path provided. Proceeding with raw values...")

        # Initialize all MEGASTRAIN5000 units
        self.imus, self.startup_config_vals = self._set_up_connected_imus(imu_ids=self.imu_ids)


    def _set_up_connected_imus(
        self,
        imu_ids: dict[int, dict[str, int]],
    ) -> tuple[list[float]]:
        """Initialize all IMUs from dictionary of IMU IDs, buses, channels, and addresses. Here you specify which IMU components to start, as well as their corresponding sensing resolution.

        Args:
            imu_ids (dict): dictionary of each IMU and the I2C bus number, multiplexer channel (if used), and I2C address needed to access it.

        Returns:
            startup_config_vals (dict of floats): ICM42607 sensor configuration values: acc_range, gyro_range, mag_coeffx, mag_coeffy, mag_coeffz.
        """
        imus = {}
        startup_config_vals = {}

        for imu_id in imu_ids.keys():
            # Get all relevant components to communicate with IMU
            bus_id = imu_ids[imu_id]['bus'] 
            address = imu_ids[imu_id]['address']

            # Initialize I2C bus if it hasn't been initialized yet
            if bus_id not in self.bus.keys():
                self.bus[bus_id] = smbus.SMBus(bus_id)

            startup_config_vals[imu_id] = {}

            # Start accelerometer and gyro if configured to do so
            if any([c for c in self.components if (('acc' in c) or ('gyro' in c))]):
                (startup_config_vals[imu_id]['acc_range'],
                startup_config_vals[imu_id]['gyro_range'],
                ) = self._set_up_ICM42607(
                        bus=self.bus[bus_id],
                        address=address,
                        imu_id = imu_id,
                        acc_range_idx=self.acc_range_selector,
                        gyro_range_idx=self.gyro_range_selector,
                )
            
            if self.verbose:
                print(f"IMU {imu_id} startup_config_vals: {startup_config_vals[imu_id]}\n")

            imus[imu_id] = IMUData()
        
        return imus, startup_config_vals

    def _set_up_ICM42607(
        self,
        bus: smbus.SMBus=smbus.SMBus(),
        address=ICM42607_ADDR,
        imu_id: int = 0,
        acc_range_idx=ACC_RANGE_8G,
        gyro_range_idx=GYRO_RANGE_500_DEG_PER_S,
        report_frequency=400,
        sleep_time=SLEEP_TIME,
    ) -> tuple[float]:
        """Set up ICM42607 integrated accelerometer and gyroscope on MEGASTRAIN5000.

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the ICM42607 unit. Default: 0x68.
            imu_id (int): IMU index, used to configure per-IMU interrupt sources. Default: 0.
            acc_range_idx (int): index for range of accelerations to collect. Default: 2 (+/- 8 g), but can be:
                0: +/- 2.0 g's
                1: +/- 4.0 g's
                2: +/- 8.0 g's
                3: +/- 16.0 g's
            gyro_range_idx (int): index for range of angular velocities to collect. Default: 2 (+/- 1000.0 deg/s), but can be:
                0: +/- 250.0 deg/s
                1: +/- 500.0 deg/s
                2: +/- 1000.0 deg/s
                3: +/- 2000.0 deg/s
            report_frequency (int): output data rate in Hz. Must be one of: 100, 200, 400, 800, 1600. Default: 400.
            sleep_time (float): time to sleep after writes. Default: 0.1 seconds.

        Returns:
            acc_range (float): +/- range of accelerometer [g].
            gyro_range (float): +/- range of gyroscope [deg/s].
        """

        # --- Step 1: Pack GYRO_CONFIG0 and ACCEL_CONFIG0 ---
        # Each register encodes FSR in bits [6:5] and ODR in bits [3:0].
        # Datasheet GYRO_CONFIG0 (0x20): FSR bits[6:5]: 00=±2000, 01=±1000, 10=±500, 11=±250
        # Datasheet ACCEL_CONFIG0 (0x21): FSR bits[6:5]: 00=±16g,  01=±8g,   10=±4g,  11=±2g
        # ODR bits[3:0]: 0101=1600Hz, 0110=800Hz, 0111=400Hz, 1000=200Hz, 1001=100Hz

        # Gyro FSR lookup (note: reversed order vs MPU9250)
        gyro_fsr_bits = [0b11, 0b10, 0b01, 0b00]  # 250, 500, 1000, 2000 dps
        gyro_config_vals = [250.0, 500.0, 1000.0, 2000.0]

        # Accel FSR lookup (note: reversed order vs MPU9250)
        acc_fsr_bits = [0b11, 0b10, 0b01, 0b00]  # 2, 4, 8, 16 g
        acc_config_vals = [2.0, 4.0, 8.0, 16.0]

        # ODR lookup
        odr_bits = {
            1600: 0b0101,
            800:  0b0110,
            400:  0b0111,
            200:  0b1000,
            100:  0b1001,
        }
        if report_frequency not in odr_bits:
            raise ValueError(f"report_frequency must be one of {list(odr_bits.keys())} Hz, got {report_frequency}")

        odr = odr_bits[report_frequency]
        gyro_byte  = (gyro_fsr_bits[gyro_range_idx] << 5) | odr
        accel_byte = (acc_fsr_bits[acc_range_idx]   << 5) | odr

        # Filter bandwidth bytes — mirror C++ choices per ODR
        # GYRO_CONFIG1 (0x23) bits[2:0]: 000=bypass, 001=180Hz, 010=121Hz, 011=73Hz, 10
        # ACCEL_CONFIG1 (0x24) bits[2:0]: same bandwidth options
        gyro_filt_byte = {1600: 0x01, 800: 0x01, 400: 0x01, 200: 0x03, 100: 0x04}[report_frequency]
        acc_filt_byte  = {1600: 0x01, 800: 0x01, 400: 0x05, 200: 0x03, 100: 0x04}[report_frequency]

        bus.write_byte_data(address, GYRO_CONFIG_0,  gyro_byte)
        time.sleep(sleep_time)
        bus.write_byte_data(address, ACCEL_CONFIG_0, accel_byte)
        time.sleep(sleep_time)
        bus.write_byte_data(address, GYRO_CONFIG_1,  gyro_filt_byte)
        time.sleep(sleep_time)
        bus.write_byte_data(address, ACCEL_CONFIG_1, acc_filt_byte)
        time.sleep(sleep_time)

        # --- Step 2: Configure interrupt pin ---
        # 0x1B = 0b00011011: pulsed mode, push-pull, active high for both INT1 and INT2
        bus.write_byte_data(address, INT_CONFIG_ICM, 0x1B)
        time.sleep(sleep_time)

        INT_SOURCE0_ICM = 0x2B
        if imu_id == 0:
            bus.write_byte_data(address, INT_SOURCE0_ICM, 0x08)
            time.sleep(sleep_time)

        # --- Step 3: Enable gyro and accel in Low Noise mode ---
        # PWR_MGMT0 (0x1F): bits[3:2]=GYRO_MODE, bits[1:0]=ACCEL_MODE
        # 0x0F = 0b00001111: gyro LN mode (11) + accel LN mode (11)
        bus.write_byte_data(address, PWR_MGMT_0, 0x0F)
        time.sleep(0.1)  # Mandatory: gyro needs minimum 45ms after power-on

        # --- Step 4: Verify WHO_AM_I ---
        # ICM42607 should return 0x60
        who_am_i = bus.read_byte_data(address, WHO_AM_I_ICM)
        if who_am_i != 0x60:
            raise RuntimeError(
                f"ICM42607 WHO_AM_I check failed at address 0x{address:02X}: "
                f"expected 0x60, got 0x{who_am_i:02X}. Check wiring and I2C address."
            )

        return acc_config_vals[acc_range_idx], gyro_config_vals[gyro_range_idx]


    def get_data(self, imu_id: int) -> IMUData:
        """Get acceleration and gyroscope data.

        Args:
            imu_id (int): IMU number (index number from starting dict, not address).

        Returns:
            imu_data (IMUData): IMU data of the current sensor.
        """
        imu_data = IMUData()
        bus = self.bus[self.imu_ids[imu_id]['bus']]
        address = self.imu_ids[imu_id]['address']
        cal_id = f"{self.imu_ids[imu_id]['bus']}_{address}"


        # Get accelerometer and gyroscope data
        if any([c for c in self.components if (('acc' in c) or ('gyro' in c))]):
            (imu_data.acc_x,
            imu_data.acc_y,
            imu_data.acc_z,
            imu_data.gyro_x,
            imu_data.gyro_y,
            imu_data.gyro_z,
            ) = self.get_ICM42607_data(
                bus=bus,
                acc_range=self.startup_config_vals[imu_id]['acc_range'],
                gyro_range=self.startup_config_vals[imu_id]['gyro_range'],
                address=address,
            )

            # If calibrations exist for current IMU, apply them
            if cal_id in self.calibration_dict.keys():
                # Calibrate accelerometer readings using a linear fit
                if len(self.calibration_dict[cal_id]["acc"]) > 0:
                    m_x = self.calibration_dict[cal_id]["acc"][0][0] # slope
                    b_x = self.calibration_dict[cal_id]["acc"][0][1] # offset
                    imu_data.acc_x = m_x * (imu_data.acc_x) + b_x

                    m_y = self.calibration_dict[cal_id]["acc"][1][0] # slope
                    b_y = self.calibration_dict[cal_id]["acc"][1][1] # offset
                    imu_data.acc_y = m_y * (imu_data.acc_y) + b_y

                    m_z = self.calibration_dict[cal_id]["acc"][2][0] # slope
                    b_z = self.calibration_dict[cal_id]["acc"][2][1] # offset
                    imu_data.acc_z = m_z * (imu_data.acc_z) + b_z

                # Calibrate gyroscope by subtracting an offset from each axis
                if len(self.calibration_dict[cal_id]["gyro"]) > 0:
                    imu_data.gyro_x = imu_data.gyro_x - self.calibration_dict[cal_id]["gyro"][0]
                    imu_data.gyro_y = imu_data.gyro_y - self.calibration_dict[cal_id]["gyro"][1]
                    imu_data.gyro_z = imu_data.gyro_z - self.calibration_dict[cal_id]["gyro"][2]


        # Update IMU data class dictionary
        imu_data.timestamp = time.perf_counter()
        self.imus[imu_id] = imu_data

        return imu_data


    def get_ICM42607_data(
        self,
        bus: smbus.SMBus,
        acc_range: float,
        gyro_range: float,
        address: int=ICM42607_ADDR,
    ) -> tuple[float]:
        """Convert raw binary accelerometer, gyroscope, and temperature readings to floats.

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            acc_range (float): +/- range of acceleration being read from ICM42607. Raw units are g's (1 g = 9.81 m*s^-2).
            gyro_range (float): +/- range of gyro being read from ICM42607. Raw units are deg/s.
            address (hex as int): address of the ICM42607 subcircuit.

        Returns:
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z: acceleration [g] and gyroscope [deg/s] values.
        """
        data = bus.read_i2c_block_data(
            i2c_addr=address,
            register=ACCEL_XOUT_H,
            length=12,
        )

        # Convert from bytes to ints
        raw_acc_x = self._convert_raw_data(data[0], data[1])
        raw_acc_y = self._convert_raw_data(data[2], data[3])
        raw_acc_z = self._convert_raw_data(data[4], data[5])
        raw_gyro_x = self._convert_raw_data(data[6], data[7])
        raw_gyro_y = self._convert_raw_data(data[8], data[9])
        raw_gyro_z = self._convert_raw_data(data[10], data[11])

        # Convert from bits to g's (accel.), deg/s (gyro), and  then 
        # from those base units to m*s^-2 and rad/s respectively
        acc_x = (raw_acc_x / (2.0**15.0)) * acc_range * GRAV_ACC
        acc_y = (raw_acc_y / (2.0**15.0)) * acc_range * GRAV_ACC
        acc_z = (raw_acc_z / (2.0**15.0)) * acc_range * GRAV_ACC
        gyro_x = (raw_gyro_x / (2.0**15.0)) * gyro_range * DEG2RAD
        gyro_y = (raw_gyro_y / (2.0**15.0)) * gyro_range * DEG2RAD
        gyro_z = (raw_gyro_z / (2.0**15.0)) * gyro_range * DEG2RAD

        return (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

    def _read_raw_bytes(
        self,
        bus: smbus.SMBus,
        address: int,
        register: int,
    ) -> int:
        raise NotImplementedError

        """Method of reading raw data from different subcircuits 
        on the MEGASTRAIN5000 board.

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the subcircuit being read from.
            register (hex as int): register from which to pull specific data.

        Returns:
            value (int): raw value pulled from specific register and converted to int.
        """
        if address == ICM42607_ADDR or address == ICM42607_ADDR_AD0_HIGH:
            # Read accel and gyro values
            high = bus.read_byte_data(address, register)
            low = bus.read_byte_data(address, register+1)
        # elif address == AK8963_ADDR:            
        #     # read magnetometer values
        #     high = bus.read_byte_data(address, register)
        #     low = bus.read_byte_data(address, register-1)

        # Combine high and low for unsigned bit value
        value = ((high << 8) | low)
        
        # Convert to +/- value
        if(value > 32768):
            value -= 65536

        return value


    def _convert_raw_data(
        self,
        high_data: int,
        low_data: int,
    ) -> int:
        # Combine high and low for unsigned bit value
        value = ((high_data << 8) | low_data)
        
        # Convert to +/- value
        if(value > 32768):
            value -= 65536

        return value


if __name__ == "__main__":
    import platform
    machine_name = platform.uname().release.lower()
    if "tegra" in machine_name:
        bus_ids = [1,7]
    elif "rpi" in machine_name or "bcm" in machine_name or "raspi" in machine_name:
        bus_ids = [15,16]
    else:
        bus_ids = [1]

    imu_dict = {
        0:
            {
                'bus': bus_ids[0],
                'address': 0x68,
            },
        1:
            {
                'bus': bus_ids[1],
                'address': 0x69,
            },
        2:
            {
                'bus': bus_ids[2],
                'address': 0x68,
            },
    }

    components = ['acc','gyro']
    verbose = True

    megastrain5000_imus = MEGASTRAIN5000IMUs(
        imu_ids=imu_dict,
        components=components,
        verbose=verbose,
    )

    loop = LoopTimer(operating_rate=160, verbose=True)
    
    while True:
        if loop.continue_loop():
            # Get data
            for imu_id in imu_dict.keys():
                imu_info = megastrain5000_imus.get_data(imu_id=imu_id)
                print(f"{imu_id}: acc_x: {imu_info.acc_x:0.2f}, acc_y: {imu_info.acc_y:0.2f}, acc_z: {imu_info.acc_z:0.2f}, gyro_x: {imu_info.gyro_x:0.2f}, gyro_y: {imu_info.gyro_y:0.2f}, gyro_z: {imu_info.gyro_z:0.2f}")