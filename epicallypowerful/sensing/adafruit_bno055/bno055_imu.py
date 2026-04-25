"""epically-powerful module for managing IMUs.

This module contains the classes and commands for initializing
and reading from BNO055 IMUs.
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

# Set BNO055 (accelerometer) registers
BNO055_CHIP_ID = 0xA0
REG_CHIP_ID = 0x00  

BNO055_ADDR = 0x29
BNO055_ADDR_AD0_LOW = 0x28
GYRO_CONFIG_0  = 0x0A
GYRO_CONFIG_1  = 0x0B
ACC_CONFIG = 0x08

INT_CONFIG_BNO = 0x0F

ACCEL_DATA_X_LSB = 0x08
ACCEL_DATA_Y_LSB = 0x0A
ACCEL_DATA_Z_LSB = 0x0C
GYRO_DATA_X_LSB = 0x14
GYRO_DATA_Y_LSB = 0x16
GYRO_DATA_Z_LSB = 0x18

# Set constants
ACC_RANGE_2G    = 0 # Set BNO055 accelerometer resolution to +/- 2 g's
ACC_RANGE_4G    = 1 # Set BNO055 accelerometer resolution to +/- 4 g's
ACC_RANGE_8G    = 2 # Set BNO055 accelerometer resolution to +/- 8 g's
ACC_RANGE_16G   = 3 # Set BNO055 accelerometer resolution to +/- 16 g's
GYRO_RANGE_250_DEG_PER_S  = 3 # Set BNO055 gyroscope resolution to +/- 250.0 deg/s
GYRO_RANGE_500_DEG_PER_S  = 2 # Set BNO055 gyroscope resolution to +/- 500.0 deg/s
GYRO_RANGE_1000_DEG_PER_S = 1 # Set BNO055 gyroscope resolution to +/- 1000.0 deg/s
GYRO_RANGE_2000_DEG_PER_S = 0 # Set BNO055 gyroscope resolution to +/- 2000.0 deg/s
SLEEP_TIME   = 0.1 # [s]
ACC_BW_7HZ          = 0b000
ACC_BW_15HZ         = 0b001
ACC_BW_31HZ         = 0b010
ACC_BW_62HZ         = 0b011  # default
ACC_BW_125HZ        = 0b100
ACC_BW_250HZ        = 0b101
ACC_BW_500HZ        = 0b110
ACC_BW_1000HZ       = 0b111
GYRO_BW_523HZ       = 0b000
GYRO_BW_230HZ       = 0b001
GYRO_BW_116HZ       = 0b010
GYRO_BW_47HZ        = 0b011  # default
GYRO_BW_23HZ        = 0b100
GYRO_BW_12HZ        = 0b101
GYRO_BW_64HZ        = 0b110
GYRO_BW_32HZ        = 0b111
ACC_PWR_NORMAL      = 0b000
GYRO_PWR_NORMAL     = 0b000

REG_PWR_MODE        = 0x3E
REG_OPR_MODE        = 0x3D
PWR_MODE_NORMAL     = 0x00
OPR_MODE_CONFIGMODE = 0x00 
OPR_MODE_NDOF       = 0x0C
OPR_MODE_ACCGYRO     = 0x05
REG_UNIT_SEL        = 0x3B
REG_PAGE_ID      = 0x07 


class BNO055IMUs(IMU):
    """Class for interfacing with the BNO055 IMU using I2C communication, leveraging the TCA9548A multiplexer for communicating with multiple units at the same time.

    This class draws from the following resources:
        - MPU9250 calibration: https://github.com/makerportal/mpu92-calibration
        - TCA9548a multiplexer to connect multiple I2C devices with the same address: https://wolles-elektronikkiste.de/en/tca9548a-i2c-multiplexer
        - TDK InvenSense MPU9250 datasheet: https://invensense.tdk.com/wp-content/uploads/2015/02/PS-MPU-9250A-01-v1.1.pdf
        - PCA9548A datasheet: https://www.ti.com/lit/ds/symlink/pca9548a.pdf
        - BNO055 datasheet: https://cdn-shop.adafruit.com/datasheets/BST_BNO055_DS000_12.pdf
        - Chase Sun's and Steven Zhou's work with the MEGASTRAIN5000 and ICM42607 IMUs

    Many helper functions are included in the :py:class:`IMUData` class to assist with getting data conveniently. Please see that documentation for all options.

    Example:
        .. code-block:: python

            from epicallypowerful.sensing import BNO055IMUs

            ### Instantiation ---
            imu_ids = {
                0: {
                    'bus': 1,
                    'channel': -1, # -1 --> no multiplexer, otherwise --> multiplexer channel
                    'address': 0x29,
                },
                1: {
                    'bus': 1,
                    'channel': -1,
                    'address': 0x28,
                },
            }

            imus = BNO055IMUs(
                imu_ids=imu_ids,
                components=['acc', 'gyro'],
            )

            ### Stream data ---
            print(imus.get_data(imu_id=0).acc_x)
            print(imus.get_data(imu_id=1).acc_x)

    Args:
        imu_ids (dict): dictionary of each IMU and the I2C bus number, multiplexer channel (if used), and I2C address needed to access it.
        components (list of strings): list of BNO055 sensing components to get. Could include `acc` or `gyro`.
        acc_range (int): index for range of accelerations to collect. Default: 2 (+/- 8 g), but can be:
            0: +/- 2.0 g's
            1: +/- 4.0 g's
            2: +/- 8.0 g's
            3: +/- 16.0 g's
        gyro_range (int): index for range of angular velocities to collect. Default: 1 (+/- 500.0 deg/s), but can be:
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
        acc_range=ACC_RANGE_8G,
        acc_bw=ACC_BW_62HZ,
        gyro_range=GYRO_RANGE_500_DEG_PER_S,
        gyro_bw=GYRO_BW_47HZ,
        opr_mode=OPR_MODE_ACCGYRO,
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
        self.acc_range = acc_range
        self.acc_bw = acc_bw
        self.gyro_range = gyro_range
        self.gyro_bw = gyro_bw
        self.verbose = verbose
        self.opr_mode = opr_mode
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

        # Initialize all BNO055 units
        self.imus, self.startup_config_vals = self._set_up_connected_imus(imu_ids=self.imu_ids)


    def _set_up_connected_imus(
        self,
        imu_ids: dict[int, dict[str, int]],
    ) -> tuple[list[float]]:
        """Initialize all IMUs from dictionary of IMU IDs, buses, channels, and addresses. Here you specify which IMU components to start, as well as their corresponding sensing resolution.

        Args:
            imu_ids (dict): dictionary of each IMU and the I2C bus number, multiplexer channel (if used), and I2C address needed to access it.

        Returns:
            startup_config_vals (dict of floats): BNO055 sensor configuration values: acc_range, gyro_range, mag_coeffx, mag_coeffy, mag_coeffz.
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
                ) = self._set_up_BNO055(
                        bus=self.bus[bus_id],
                        address=address,
                        imu_id = imu_id,
                        acc_range_idx=self.acc_range,
                        gyro_range_idx=self.gyro_range,
                )
            
            if self.verbose:
                print(f"IMU {imu_id} startup_config_vals: {startup_config_vals[imu_id]}\n")

            imus[imu_id] = IMUData()
        
        return imus, startup_config_vals

    def _set_up_BNO055(
        self,
        bus: smbus.SMBus=smbus.SMBus(),
        address=BNO055_ADDR,
        imu_id: int = 0,
        acc_range_idx=ACC_RANGE_8G,
        gyro_range_idx=GYRO_RANGE_500_DEG_PER_S,
        report_frequency=400,
        sleep_time=SLEEP_TIME,
    ) -> tuple[float]:
        """Initialise a single BNO055 unit.
 
        Sequence (per datasheet):
            1. Verify CHIP_ID (reg 0x00 = 0xA0).
            2. Set PWR_MODE to Normal.
            3. Enter CONFIGMODE.
            4. Write UNIT_SEL for m/s² and deg/s.
            5. Switch to Page 1.
            6. Write ACC_Config and GYR_Config_0/1.
            7. Switch back to Page 0.
            8. Enter the requested operating mode.
 
        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the BNO055 unit. Default: 0x29.
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
        chip_id = bus.read_byte_data(address, REG_CHIP_ID)
        if chip_id != BNO055_CHIP_ID:
            raise RuntimeError(
                f"BNO055 CHIP_ID check failed for IMU {imu_id} at address "
                f"0x{address:02X}: expected 0x{BNO055_CHIP_ID:02X}, "
                f"got 0x{chip_id:02X}. Check wiring and I2C address."
            )
        if self.verbose:
            print(f"IMU {imu_id}: CHIP_ID OK (0x{chip_id:02X})")
 
        bus.write_byte_data(address, REG_PWR_MODE, PWR_MODE_NORMAL)
        time.sleep(sleep_time)
 
        # Enter CONFIGMODE — required before writing any config reg
        bus.write_byte_data(address, REG_OPR_MODE, OPR_MODE_CONFIGMODE)
        time.sleep(sleep_time)
 
        # Configure output units — m/s² for accel, deg/s for gyro,
        # Android orientation  
        # UNIT_SEL = 0x00: accel m/s² (bit0=0), gyro dps (bit1=0),
        #                   Euler deg (bit2=0), temp °C (bit4=0),
        #                   Android orientation (bit7=0)
        bus.write_byte_data(address, REG_UNIT_SEL, 0x00)
        time.sleep(sleep_time)
 
        # Switch to Page 1 to access sensor config registers
        bus.write_byte_data(address, REG_PAGE_ID, 0x01)
        time.sleep(sleep_time)
 
        # Write ACC_Config 
        #   bits [1:0] = ACC_Range
        #   bits [4:2] = ACC_BW
        #   bits [7:5] = ACC_PWR_Mode (Normal = 0b000)
        acc_config_byte = (
            (ACC_PWR_NORMAL    << 5) |
            (self.acc_bw       << 2) |
            (self.acc_range    << 0)
        )
        bus.write_byte_data(address, ACC_CONFIG, acc_config_byte)
        time.sleep(sleep_time)
 
        # Write GYRO_Config_0  
        #   bits [2:0] = GYR_Range
        #   bits [5:3] = GYR_Bandwidth
        gyr_config0_byte = (
            (self.gyro_bw    << 3) |
            (self.gyro_range << 0)
        )
        bus.write_byte_data(address, GYRO_CONFIG_0, gyr_config0_byte)
        time.sleep(sleep_time)
 
        # Write GYRO_Config_1  
        #   bits [2:0] = GYRO_Power_Mode (Normal = 0b000)
        bus.write_byte_data(address, GYRO_CONFIG_1, GYRO_PWR_NORMAL)
        time.sleep(sleep_time)
 
        # Switch back to Page 0 for data registers
        bus.write_byte_data(address, REG_PAGE_ID, 0x00)
        time.sleep(sleep_time)
 
        # Enter the requested operating mode
        bus.write_byte_data(address, REG_OPR_MODE, self.opr_mode)
        time.sleep(sleep_time)
 
        if self.verbose:
            print(
                f"IMU {imu_id}: ACC_Config=0x{acc_config_byte:02X}  "
                f"GYR_Config0=0x{gyr_config0_byte:02X}  "
                f"OPR_MODE=0x{self.opr_mode:02X}"
            )

        return acc_range_idx, gyro_range_idx


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
            ) = self.get_BNO055_data(
                bus=bus,
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


    def get_BNO055_data(
        self,
        bus: smbus.SMBus,
        address: int=BNO055_ADDR,
    ) -> tuple[float]:
        """Convert raw binary accelerometer, gyroscope, and temperature readings to floats.

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the BNO055 subcircuit.

        Returns:
            acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z: acceleration [g] and gyroscope [deg/s] values.
        """
        data = bus.read_i2c_block_data(
            i2c_addr=address,
            register=ACCEL_DATA_X_LSB,
            length=18,
        )

        # Convert from bytes to ints
        raw_acc_x = self._convert_raw_data(data[0], data[1])
        raw_acc_y = self._convert_raw_data(data[2], data[3])
        raw_acc_z = self._convert_raw_data(data[4], data[5])
        raw_gyro_x = self._convert_raw_data(data[12], data[13])
        raw_gyro_y = self._convert_raw_data(data[14], data[15])
        raw_gyro_z = self._convert_raw_data(data[16], data[17])

        # Convert from bits to g's (accel.), deg/s (gyro), and  then 
        # from those base units to m*s^-2 and rad/s respectively
        acc_x  = raw_acc_x  / 100.0                     # 100 LSB per m/s²
        acc_y  = raw_acc_y  / 100.0
        acc_z  = raw_acc_z  / 100.0
        gyro_x = (raw_gyro_x / 16.0) * DEG2RAD          # 16 LSB per dps → rad/s
        gyro_y = (raw_gyro_y / 16.0) * DEG2RAD
        gyro_z = (raw_gyro_z / 16.0) * DEG2RAD

        return (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

    def _read_raw_bytes(
        self,
        bus: smbus.SMBus,
        address: int,
        register: int,
    ) -> int:
        raise NotImplementedError

        """Method of reading raw data from different subcircuits 
        on the BNO055 board.

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the subcircuit being read from.
            register (hex as int): register from which to pull specific data.

        Returns:
            value (int): raw value pulled from specific register and converted to int.
        """
        if address == BNO055_ADDR or address == BNO055_ADDR_AD0_HIGH:
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
        if(value > 32767):
            value -= 65536

        return value


    def _convert_raw_data(
        self,
        low_data: int,
        high_data: int,
    ) -> int:
        # Combine high and low for unsigned bit value
        value = ((high_data << 8) | low_data)
        
        # Convert to +/- value
        if(value > 32767):
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
                'address': 0x29,
            },
        1:
            {
                'bus': bus_ids[1],
                'address': 0x29,
            },
        2:
            {
                'bus': bus_ids[2],
                'address': 0x29,
            },
    }

    components = ['acc','gyro']
    verbose = True

    bno055_imus = BNO055IMUs(
        imu_ids=imu_dict,
        components=components,
        acc_range=ACC_RANGE_8G,
        gyro_range=GYRO_RANGE_500_DEG_PER_S,
        acc_bw=ACC_BW_62HZ,
        gyro_bw=GYRO_BW_47HZ,
        opr_mode=OPR_MODE_ACCGYRO,
        verbose=verbose,
    )

    loop = LoopTimer(operating_rate=100, verbose=True)
    
    while True:
        if loop.continue_loop():
            # Get data
            for imu_id in imu_dict.keys():
                imu_info = bno055_imus.get_data(imu_id=imu_id)
                print(f"{imu_id}: acc_x: {imu_info.acc_x:0.2f}, acc_y: {imu_info.acc_y:0.2f}, acc_z: {imu_info.acc_z:0.2f}, gyro_x: {imu_info.gyro_x:0.2f}, gyro_y: {imu_info.gyro_y:0.2f}, gyro_z: {imu_info.gyro_z:0.2f}")