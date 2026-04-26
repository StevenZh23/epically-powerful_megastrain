"""epically-powerful module for managing IMUs.

This module contains the classes and commands for initializing
and reading from LSM6DSOX IMUs.
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

# LSM6DSOX identity  (datasheet section 9.14, p.53)
# WHO_AM_I is at register 0x0F, fixed value 0x6C
LSM6DSOX_CHIP_ID        = 0x6C
REG_WHO_AM_I            = 0x0F

# I2C addresses  (datasheet section 10.2, p.xx)
# SAD = 110101xb; SDO/SA0 pin selects LSb
# SDO/SA0 HIGH → 1101011b = 0x6B (default when pin pulled high)
# SDO/SA0 LOW  → 1101010b = 0x6A
LSM6DSOX_ADDR           = 0x6B  # default (SDO/SA0 HIGH)
LSM6DSOX_ADDR_SA0_LOW   = 0x6A  # alternate (SDO/SA0 LOW)

# Control registers  (datasheet Table 19 register map, p.43)
REG_CTRL1_XL            = 0x10  # Accel ODR, FS
REG_CTRL2_G             = 0x11  # Gyro ODR, FS
REG_CTRL3_C             = 0x12  # SW reset, IF_INC, BDU
REG_CTRL6_C             = 0x15  # Accel high-performance mode
REG_CTRL7_G             = 0x16  # Gyro high-performance mode

# Output data registers — little-endian (LSB at lower address)
# (datasheet register map, p.43-44)
GYRO_DATA_X_LSB         = 0x22  # Gyro X LSB  (OUTX_L_G)
GYRO_DATA_Y_LSB         = 0x24  # Gyro Y LSB
GYRO_DATA_Z_LSB         = 0x26  # Gyro Z LSB
ACCEL_DATA_X_LSB        = 0x28  # Accel X LSB (OUTX_L_A)
ACCEL_DATA_Y_LSB        = 0x2A  # Accel Y LSB
ACCEL_DATA_Z_LSB        = 0x2C  # Accel Z LSB

# ---------------------------------------------------------------------------
# Accelerometer full-scale (FS_XL) encoding  (datasheet Table 52, p.54)
# Written to CTRL1_XL bits [3:2]  (XL_FS_MODE=0 in CTRL8_XL, default)
#   00 → ±2 g    sensitivity: 0.061 mg/LSB
#   01 → ±16 g   sensitivity: 0.488 mg/LSB
#   10 → ±4 g    sensitivity: 0.122 mg/LSB
#   11 → ±8 g    sensitivity: 0.244 mg/LSB
# ---------------------------------------------------------------------------
ACC_RANGE_2G            = 0b00  # ±2 g
ACC_RANGE_16G           = 0b01  # ±16 g
ACC_RANGE_4G            = 0b10  # ±4 g
ACC_RANGE_8G            = 0b11  # ±8 g  ← default

# Sensitivity in mg/LSB for each FS_XL setting  (datasheet Table 3, p.xx)
ACC_SENSITIVITY_MG_PER_LSB = {
    ACC_RANGE_2G:   0.061,
    ACC_RANGE_16G:  0.488,
    ACC_RANGE_4G:   0.122,
    ACC_RANGE_8G:   0.244,
}

# ---------------------------------------------------------------------------
# Gyroscope full-scale (FS_G) encoding  (datasheet Table 54, p.55)
# Written to CTRL2_G bits [3:2]; FS_125 is bit [1]
#   00 → ±250 dps   sensitivity: 8.75  mdps/LSB
#   01 → ±500 dps   sensitivity: 17.50 mdps/LSB
#   10 → ±1000 dps  sensitivity: 35    mdps/LSB
#   11 → ±2000 dps  sensitivity: 70    mdps/LSB
# FS_125 bit set → ±125 dps, sensitivity: 4.375 mdps/LSB
# NOTE: LSM6DSOX uses ±250 dps (not ±245 dps as on the LSM6DS3TR-C)
# ---------------------------------------------------------------------------
GYRO_RANGE_125_DEG_PER_S  = 4  # special value — triggers FS_125 bit; not a direct FS_G field value
GYRO_RANGE_250_DEG_PER_S  = 0b00  # FS_G=00
GYRO_RANGE_500_DEG_PER_S  = 0b01  # FS_G=01
GYRO_RANGE_1000_DEG_PER_S = 0b10  # FS_G=10
GYRO_RANGE_2000_DEG_PER_S = 0b11  # FS_G=11  ← default

# Sensitivity in mdps/LSB for each gyro range  (datasheet Table 3)
GYRO_SENSITIVITY_MDPS_PER_LSB = {
    GYRO_RANGE_125_DEG_PER_S:  4.375,
    GYRO_RANGE_250_DEG_PER_S:  8.75,
    GYRO_RANGE_500_DEG_PER_S:  17.50,
    GYRO_RANGE_1000_DEG_PER_S: 35.0,
    GYRO_RANGE_2000_DEG_PER_S: 70.0,
}

# ---------------------------------------------------------------------------
# Accelerometer ODR encoding, CTRL1_XL bits [7:4]  (datasheet Table 51, p.54)
# High-performance mode (XL_HM_MODE=0 in CTRL6_C, which is the default)
# ---------------------------------------------------------------------------
ACC_ODR_POWER_DOWN      = 0b0000
ACC_ODR_12_5HZ          = 0b0001  # 12.5 Hz high-performance
ACC_ODR_26HZ            = 0b0010  # 26 Hz
ACC_ODR_52HZ            = 0b0011  # 52 Hz
ACC_ODR_104HZ           = 0b0100  # 104 Hz  ← default
ACC_ODR_208HZ           = 0b0101  # 208 Hz
ACC_ODR_416HZ           = 0b0110  # 416 Hz
ACC_ODR_833HZ           = 0b0111  # 833 Hz
ACC_ODR_1660HZ          = 0b1000  # 1.66 kHz
ACC_ODR_3330HZ          = 0b1001  # 3.33 kHz
ACC_ODR_6660HZ          = 0b1010  # 6.66 kHz

# ---------------------------------------------------------------------------
# Gyroscope ODR encoding, CTRL2_G bits [7:4]  (datasheet Table 55, p.55)
# High-performance mode (G_HM_MODE=0 in CTRL7_G, which is the default)
# ---------------------------------------------------------------------------
GYRO_ODR_POWER_DOWN     = 0b0000
GYRO_ODR_12_5HZ         = 0b0001  # 12.5 Hz
GYRO_ODR_26HZ           = 0b0010  # 26 Hz
GYRO_ODR_52HZ           = 0b0011  # 52 Hz
GYRO_ODR_104HZ          = 0b0100  # 104 Hz  ← default
GYRO_ODR_208HZ          = 0b0101  # 208 Hz
GYRO_ODR_416HZ          = 0b0110  # 416 Hz
GYRO_ODR_833HZ          = 0b0111  # 833 Hz
GYRO_ODR_1660HZ         = 0b1000  # 1.66 kHz
GYRO_ODR_3330HZ         = 0b1001  # 3.33 kHz
GYRO_ODR_6660HZ         = 0b1010  # 6.66 kHz

SLEEP_TIME   = 0.1 # [s]


class LSM6DSOXIMUs(IMU):
    """Class for interfacing with the LSM6DSOX IMU using I2C communication, leveraging the TCA9548A multiplexer for communicating with multiple units at the same time.

    This class draws from the following resources:
        - MPU9250 calibration: https://github.com/makerportal/mpu92-calibration
        - TCA9548a multiplexer to connect multiple I2C devices with the same address: https://wolles-elektronikkiste.de/en/tca9548a-i2c-multiplexer
        - LSM6DSOX datasheet: https://www.st.com/resource/en/datasheet/lsm6dsox.pdf
        - Chase Sun's and Steven Zhou's work with the MEGASTRAIN5000 and ICM42607 IMUs

    Many helper functions are included in the :py:class:`IMUData` class to assist with getting data conveniently. Please see that documentation for all options.

    Example:
        .. code-block:: python

            from epicallypowerful.sensing import LSM6DSOXIMUs

            ### Instantiation ---
            imu_ids = {
                0: {
                    'bus': 1,
                    'channel': -1, # -1 --> no multiplexer, otherwise --> multiplexer channel
                    'address': 0x6B,
                },
                1: {
                    'bus': 1,
                    'channel': -1,
                    'address': 0x6A,
                },
            }

            imus = LSM6DSOXIMUs(
                imu_ids=imu_ids,
                components=['acc', 'gyro'],
            )

            ### Stream data ---
            print(imus.get_data(imu_id=0).acc_x)
            print(imus.get_data(imu_id=1).acc_x)

    Args:
        imu_ids (dict): dictionary of each IMU and the I2C bus number, multiplexer channel (if used), and I2C address needed to access it.
        components (list of strings): list of LSM6DSOX sensing components to get. Could include `acc` or `gyro`.
        acc_range (int): full-scale selection for accelerometer. Default: ACC_RANGE_8G, but can be:
            ACC_RANGE_2G:  ±2 g  (0.061 mg/LSB)
            ACC_RANGE_4G:  ±4 g  (0.122 mg/LSB)
            ACC_RANGE_8G:  ±8 g  (0.244 mg/LSB)
            ACC_RANGE_16G: ±16 g (0.488 mg/LSB)
        acc_odr (int): output data rate for accelerometer. Default: ACC_ODR_104HZ.
        gyro_range (int): full-scale selection for gyroscope. Default: GYRO_RANGE_2000_DEG_PER_S, but can be:
            GYRO_RANGE_125_DEG_PER_S:  ±125 dps  (4.375 mdps/LSB)
            GYRO_RANGE_250_DEG_PER_S:  ±250 dps  (8.75  mdps/LSB)
            GYRO_RANGE_500_DEG_PER_S:  ±500 dps  (17.50 mdps/LSB)
            GYRO_RANGE_1000_DEG_PER_S: ±1000 dps (35    mdps/LSB)
            GYRO_RANGE_2000_DEG_PER_S: ±2000 dps (70    mdps/LSB)
        gyro_odr (int): output data rate for gyroscope. Default: GYRO_ODR_104HZ.
        calibration_path (str): path to JSON file with calibration values for IMUs to be connected. NOTE: this file indexes IMUs by which bus, multiplexer channel (if used), and I2C address they are connected to. Be careful not to use the calibration for one IMU connected in this way on another unit by mistake.
        verbose (bool): whether to print verbose output from IMU operation. Default: False.
    """

    def __init__(
        self,
        imu_ids: dict[int, dict[str, int]],
        components=['acc','gyro'],
        acc_range=ACC_RANGE_8G,
        acc_odr=ACC_ODR_104HZ,
        gyro_range=GYRO_RANGE_2000_DEG_PER_S,
        gyro_odr=GYRO_ODR_104HZ,
        calibration_path='',
        verbose: bool=False,
    ) -> None:
        if imu_ids is None:
            raise Exception('`imu_ids` must contain at least one IMU index.')
        elif not isinstance(imu_ids,dict):
            raise Exception ('`imu_ids` must be in the form of dict(int, dict(int bus_id, hex imu_id).')

        # Initialize all IMU-specific class attributes
        self.imu_ids = imu_ids
        self.components = components
        self.acc_range = acc_range
        self.acc_odr = acc_odr
        self.gyro_range = gyro_range
        self.gyro_odr = gyro_odr
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

        # Initialize all LSM6DSOX units
        self.imus, self.startup_config_vals = self._set_up_connected_imus(imu_ids=self.imu_ids)


    def _set_up_connected_imus(
        self,
        imu_ids: dict[int, dict[str, int]],
    ) -> tuple[list[float]]:
        """Initialize all IMUs from dictionary of IMU IDs, buses, channels, and addresses.

        Args:
            imu_ids (dict): dictionary of each IMU and the I2C bus number, multiplexer channel (if used), and I2C address needed to access it.

        Returns:
            startup_config_vals (dict of floats): LSM6DSOX sensor configuration values: acc_range, gyro_range.
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
                ) = self._set_up_LSM6DSOX(
                        bus=self.bus[bus_id],
                        address=address,
                        imu_id=imu_id,
                        acc_range_idx=self.acc_range,
                        gyro_range_idx=self.gyro_range,
                )
            
            if self.verbose:
                print(f"IMU {imu_id} startup_config_vals: {startup_config_vals[imu_id]}\n")

            imus[imu_id] = IMUData()
        
        return imus, startup_config_vals

    def _set_up_LSM6DSOX(
        self,
        bus: smbus.SMBus=smbus.SMBus(),
        address=LSM6DSOX_ADDR,
        imu_id: int=0,
        acc_range_idx=ACC_RANGE_8G,
        gyro_range_idx=GYRO_RANGE_2000_DEG_PER_S,
        sleep_time=SLEEP_TIME,
    ) -> tuple[float]:
        """Initialise a single LSM6DSOX unit.

        Sequence (per datasheet):
            1. Verify WHO_AM_I (reg 0x0F = 0x6C).
            2. Software reset via CTRL3_C SW_RESET bit; wait for it to clear.
            3. Write CTRL3_C: enable IF_INC (auto address increment) and BDU
               (block data update — output registers not updated until both
               MSB and LSB have been read).
            4. Write CTRL1_XL: set accel ODR and FS.
            5. Write CTRL2_G:  set gyro  ODR and FS.
            6. Write CTRL6_C:  keep accel in high-performance mode (XL_HM_MODE=0).
            7. Write CTRL7_G:  keep gyro  in high-performance mode (G_HM_MODE=0).

        Notes:
            Unlike the BNO055, the LSM6DSOX has no separate config/run mode.
            All registers are writable at any time. There is also no separate
            page system — all registers share a single flat address space.

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the LSM6DSOX unit. Default: 0x6B.
            imu_id (int): IMU index, used for log messages. Default: 0.
            acc_range_idx (int): full-scale selection for accelerometer. Default: ACC_RANGE_8G.
            gyro_range_idx (int): full-scale selection for gyroscope. Default: GYRO_RANGE_2000_DEG_PER_S.
            sleep_time (float): time to sleep after writes. Default: 0.1 seconds.

        Returns:
            acc_range_idx (int): accelerometer FS index as passed in.
            gyro_range_idx (int): gyroscope FS index as passed in.
        """
        # Step 1: Verify WHO_AM_I  (datasheet section 9.14, p.53; expected 0x6C)
        chip_id = bus.read_byte_data(address, REG_WHO_AM_I)
        if chip_id != LSM6DSOX_CHIP_ID:
            raise RuntimeError(
                f"LSM6DSOX WHO_AM_I check failed for IMU {imu_id} at address "
                f"0x{address:02X}: expected 0x{LSM6DSOX_CHIP_ID:02X}, "
                f"got 0x{chip_id:02X}. Check wiring and I2C address."
            )
        if self.verbose:
            print(f"IMU {imu_id}: WHO_AM_I OK (0x{chip_id:02X})")
 
        # Step 2: Software reset  (datasheet section 9.17 CTRL3_C, p.56)
        # SW_RESET = bit 0; self-clearing once reset is complete
        bus.write_byte_data(address, REG_CTRL3_C, 0x01)
        time.sleep(sleep_time)

        # Step 3: Write CTRL3_C — enable IF_INC (bit 2) and BDU (bit 6)
        # IF_INC=1: register address auto-increments during block reads
        # BDU=1:    output register pair not updated until both bytes read
        # Note: IF_INC defaults to 1 on LSM6DSOX but we set it explicitly
        # (datasheet section 9.17, p.56)
        ctrl3_c = (1 << 6) | (1 << 2)  # BDU=1, IF_INC=1
        bus.write_byte_data(address, REG_CTRL3_C, ctrl3_c)
        time.sleep(sleep_time)

        # Step 4: Write CTRL1_XL — accel ODR [7:4] and FS_XL [3:2]
        # (datasheet section 9.15, p.54; Tables 50-52)
        # bit layout: ODR_XL[3:0] | FS[1:0]_XL | LPF2_XL_EN | 0
        ctrl1_xl = (self.acc_odr << 4) | (acc_range_idx << 2)
        bus.write_byte_data(address, REG_CTRL1_XL, ctrl1_xl)
        time.sleep(sleep_time)

        # Step 5: Write CTRL2_G — gyro ODR [7:4] and FS_G [3:2]
        # Special case: ±125 dps uses FS_G=00 with FS_125 bit (bit 1) set.
        # All other ranges use FS_125=0 and FS_G[1:0] directly.
        # (datasheet section 9.16, p.55; Tables 53-55)
        # bit layout: ODR_G[3:0] | FS[1:0]_G | 0 | FS_125
        if gyro_range_idx == GYRO_RANGE_125_DEG_PER_S:
            ctrl2_g = (self.gyro_odr << 4) | (0b00 << 2) | (1 << 1)  # FS_125=1
        else:
            ctrl2_g = (self.gyro_odr << 4) | (gyro_range_idx << 2)
        bus.write_byte_data(address, REG_CTRL2_G, ctrl2_g)
        time.sleep(sleep_time)

        # Step 6: Write CTRL6_C — keep accel in high-performance mode
        # XL_HM_MODE = bit 4; 0 = high-performance enabled (default)
        # (datasheet section 9.20, p.59)
        bus.write_byte_data(address, REG_CTRL6_C, 0x00)
        time.sleep(sleep_time)

        # Step 7: Write CTRL7_G — keep gyro in high-performance mode
        # G_HM_MODE = bit 7; 0 = high-performance enabled (default)
        # (datasheet section 9.21, p.59-60)
        bus.write_byte_data(address, REG_CTRL7_G, 0x00)
        time.sleep(sleep_time)

        if self.verbose:
            print(
                f"IMU {imu_id}: CTRL1_XL=0x{ctrl1_xl:02X}  "
                f"CTRL2_G=0x{ctrl2_g:02X}  "
                f"CTRL3_C=0x{ctrl3_c:02X}"
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
            ) = self.get_LSM6DSOX_data(
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


    def get_LSM6DSOX_data(
        self,
        bus: smbus.SMBus,
        address: int=LSM6DSOX_ADDR,
    ) -> tuple[float]:
        """Convert raw binary accelerometer and gyroscope readings to floats.

        Reads 12 consecutive bytes starting at OUTX_L_G (0x22):
            bytes 0-1:   gyro  X (LSB, MSB)
            bytes 2-3:   gyro  Y (LSB, MSB)
            bytes 4-5:   gyro  Z (LSB, MSB)
            bytes 6-7:   accel X (LSB, MSB)
            bytes 8-9:   accel Y (LSB, MSB)
            bytes 10-11: accel Z (LSB, MSB)

        Requires IF_INC=1 in CTRL3_C (set during init) for the block read
        to auto-increment through consecutive registers.

        Sensitivity scaling  (datasheet Table 3):
            Accel: mg/LSB × 9.81/1000  → m/s²   (FS-dependent)
            Gyro:  mdps/LSB / 1000     → dps     (FS-dependent)
                   then × DEG2RAD      → rad/s

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the LSM6DSOX unit.

        Returns:
            acc_x, acc_y, acc_z [m/s²], gyro_x, gyro_y, gyro_z [rad/s]
        """
        # Read 12 bytes: gyro X/Y/Z (6 bytes) then accel X/Y/Z (6 bytes)
        data = bus.read_i2c_block_data(
            i2c_addr=address,
            register=GYRO_DATA_X_LSB,  # start at OUTX_L_G (0x22)
            length=12,
        )

        # Convert from bytes to signed 16-bit ints (little-endian)
        raw_gyro_x = self._convert_raw_data(data[0],  data[1])
        raw_gyro_y = self._convert_raw_data(data[2],  data[3])
        raw_gyro_z = self._convert_raw_data(data[4],  data[5])
        raw_acc_x  = self._convert_raw_data(data[6],  data[7])
        raw_acc_y  = self._convert_raw_data(data[8],  data[9])
        raw_acc_z  = self._convert_raw_data(data[10], data[11])

        # Accel: raw × (mg/LSB) × (m/s² per mg)  →  m/s²
        acc_sens = ACC_SENSITIVITY_MG_PER_LSB[self.acc_range]
        acc_x = raw_acc_x * acc_sens * GRAV_ACC / 1000.0
        acc_y = raw_acc_y * acc_sens * GRAV_ACC / 1000.0
        acc_z = raw_acc_z * acc_sens * GRAV_ACC / 1000.0

        # Gyro: raw × (mdps/LSB) / 1000  →  dps  →  × DEG2RAD  →  rad/s
        gyro_sens = GYRO_SENSITIVITY_MDPS_PER_LSB.get(self.gyro_range, 70.0)
        gyro_x = (raw_gyro_x * gyro_sens / 1000.0) * DEG2RAD
        gyro_y = (raw_gyro_y * gyro_sens / 1000.0) * DEG2RAD
        gyro_z = (raw_gyro_z * gyro_sens / 1000.0) * DEG2RAD

        return (acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)

    def _read_raw_bytes(
        self,
        bus: smbus.SMBus,
        address: int,
        register: int,
    ) -> int:
        raise NotImplementedError

        """Method of reading raw data from the LSM6DSOX board.

        Args:
            bus (smbus.SMBus): I2C bus instance on the device.
            address (hex as int): address of the subcircuit being read from.
            register (hex as int): register from which to pull specific data.

        Returns:
            value (int): raw value pulled from specific register and converted to int.
        """
        if address == LSM6DSOX_ADDR or address == LSM6DSOX_ADDR_SA0_LOW:
            # Read accel and gyro values (little-endian: LSB first)
            low  = bus.read_byte_data(address, register)
            high = bus.read_byte_data(address, register+1)

        # Combine low and high for unsigned bit value
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
        # Combine low and high for unsigned bit value (little-endian)
        value = ((high_data << 8) | low_data)
        
        # Convert to +/- value
        if(value > 32767):
            value -= 65536

        return value


if __name__ == "__main__":
    import platform
    machine_name = platform.uname().release.lower()
    if "tegra" in machine_name:
        bus_ids = [1, 7]
    elif "rpi" in machine_name or "bcm" in machine_name or "raspi" in machine_name:
        bus_ids = [15, 16]
    else:
        bus_ids = [1]

    imu_dict = {
        0:
            {
                'bus': bus_ids[0],
                'address': LSM6DSOX_ADDR,
            },
    }

    components = ['acc', 'gyro']
    verbose = True

    lsm6dsox_imus = LSM6DSOXIMUs(
        imu_ids=imu_dict,
        components=components,
        acc_range=ACC_RANGE_8G,
        acc_odr=ACC_ODR_104HZ,
        gyro_range=GYRO_RANGE_2000_DEG_PER_S,
        gyro_odr=GYRO_ODR_104HZ,
        verbose=verbose,
    )

    loop = LoopTimer(operating_rate=104, verbose=True)  # 104 Hz = default ODR
    
    while True:
        if loop.continue_loop():
            for imu_id in imu_dict.keys():
                imu_info = lsm6dsox_imus.get_data(imu_id=imu_id)
                print(f"{imu_id}: acc_x: {imu_info.acc_x:0.2f}, acc_y: {imu_info.acc_y:0.2f}, acc_z: {imu_info.acc_z:0.2f}, gyro_x: {imu_info.gyro_x:0.2f}, gyro_y: {imu_info.gyro_y:0.2f}, gyro_z: {imu_info.gyro_z:0.2f}")
