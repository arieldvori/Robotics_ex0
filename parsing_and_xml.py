import csv
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import navpy
from gnssutils import EphemerisManager
import simplekml
import warnings

# Filter out future warnings
warnings.filterwarnings("ignore", category=FutureWarning)

LIGHTSPEED = 2.99792458e8
WEEKSEC = 604800


def open_file(path: str):
    with open(path) as csvfile:
        measurements = []
        android_fixes = []
        reader = csv.reader(csvfile)
        for row in reader:
            if row[0][0] == '#':
                if 'Fix' in row[0]:
                    android_fixes = [row[1:]]
                elif 'Raw' in row[0]:
                    measurements = [row[1:]]
            else:
                if row[0] == 'Fix':
                    android_fixes.append(row[1:])
                elif row[0] == 'Raw':
                    measurements.append(row[1:])

    measurements = pd.DataFrame(measurements[1:], columns=measurements[0])
    android_fixes = pd.DataFrame(android_fixes[1:], columns=android_fixes[0])

    # Format satellite IDs
    measurements.loc[measurements['Svid'].str.len() == 1, 'Svid'] = '0' + measurements['Svid']
    measurements.loc[measurements['ConstellationType'] == '1', 'Constellation'] = 'G'
    measurements.loc[measurements['ConstellationType'] == '3', 'Constellation'] = 'R'
    measurements['SvName'] = measurements['Constellation'] + measurements['Svid']

    # Remove all non-GPS measurements
    measurements = measurements.loc[measurements['Constellation'] == 'G']

    # Convert columns to numeric representation
    measurements['Cn0DbHz'] = pd.to_numeric(measurements['Cn0DbHz'])
    measurements['TimeNanos'] = pd.to_numeric(measurements['TimeNanos'])
    measurements['FullBiasNanos'] = pd.to_numeric(measurements['FullBiasNanos'])
    measurements['ReceivedSvTimeNanos'] = pd.to_numeric(measurements['ReceivedSvTimeNanos'])
    measurements['PseudorangeRateMetersPerSecond'] = pd.to_numeric(measurements['PseudorangeRateMetersPerSecond'])
    measurements['ReceivedSvTimeUncertaintyNanos'] = pd.to_numeric(measurements['ReceivedSvTimeUncertaintyNanos'])

    # A few measurement values are not provided by all phones
    # We'll check for them and initialize them with zeros if missing
    if 'BiasNanos' in measurements.columns:
        measurements['BiasNanos'] = pd.to_numeric(measurements['BiasNanos'])
    else:
        measurements['BiasNanos'] = 0
    if 'TimeOffsetNanos' in measurements.columns:
        measurements['TimeOffsetNanos'] = pd.to_numeric(measurements['TimeOffsetNanos'])
    else:
        measurements['TimeOffsetNanos'] = 0

    measurements['GpsTimeNanos'] = measurements['TimeNanos'] - (
            measurements['FullBiasNanos'] - measurements['BiasNanos'])
    gpsepoch = datetime(1980, 1, 6, 0, 0, 0)
    measurements['UnixTime'] = pd.to_datetime(measurements['GpsTimeNanos'], utc=True, origin=gpsepoch)
    measurements['UnixTime'] = measurements['UnixTime']

    # Split data into measurement epochs
    measurements['Epoch'] = 0
    measurements.loc[
        measurements['UnixTime'] - measurements['UnixTime'].shift() > timedelta(milliseconds=200), 'Epoch'] = 1
    measurements['Epoch'] = measurements['Epoch'].cumsum()

    # This should account for rollovers since it uses a week number specific to each measurement

    measurements['tRxGnssNanos'] = measurements['TimeNanos'] + measurements['TimeOffsetNanos'] - (
            measurements['FullBiasNanos'].iloc[0] + measurements['BiasNanos'].iloc[0])
    measurements['GpsWeekNumber'] = np.floor(1e-9 * measurements['tRxGnssNanos'] / WEEKSEC)
    measurements['tRxSeconds'] = 1e-9 * measurements['tRxGnssNanos'] - WEEKSEC * measurements['GpsWeekNumber']
    measurements['tTxSeconds'] = 1e-9 * (measurements['ReceivedSvTimeNanos'] + measurements['TimeOffsetNanos'])
    # Calculate pseudorange in seconds
    measurements['prSeconds'] = measurements['tRxSeconds'] - measurements['tTxSeconds']

    # Conver to meters
    measurements['PrM'] = LIGHTSPEED * measurements['prSeconds']
    measurements['PrSigmaM'] = LIGHTSPEED * 1e-9 * measurements['ReceivedSvTimeUncertaintyNanos']

    return measurements, android_fixes


def calculate_satellite_position(ephemeris, timestamp, one_epoch) -> pd.DataFrame:
    mu = 3.986005e14
    OmegaDot_e = 7.2921151467e-5
    F = -4.442807633e-10
    sv_position = pd.DataFrame()
    sv_position['sv'] = ephemeris.index
    sv_position.set_index('sv', inplace=True)
    sv_position['GPS time'] = timestamp  # hhhhh
    transmit_time = one_epoch['tTxSeconds']  # gggggg
    sv_position['t_k'] = transmit_time - ephemeris['t_oe']
    A = ephemeris['sqrtA'].pow(2)
    n_0 = np.sqrt(mu / A.pow(3))
    n = n_0 + ephemeris['deltaN']
    M_k = ephemeris['M_0'] + n * sv_position['t_k']
    E_k = M_k
    err = pd.Series(data=[1] * len(sv_position.index))
    i = 0
    while err.abs().min() > 1e-8 and i < 10:
        new_vals = M_k + ephemeris['e'] * np.sin(E_k)
        err = new_vals - E_k
        E_k = new_vals
        i += 1

    sinE_k = np.sin(E_k)
    cosE_k = np.cos(E_k)
    delT_oc = transmit_time - ephemeris['t_oc']
    sv_position['delT_sv'] = ephemeris['SVclockBias'] + ephemeris['SVclockDrift'] * delT_oc + ephemeris[
        'SVclockDriftRate'] * delT_oc.pow(2)
    pr = one_epoch['PrM'] + LIGHTSPEED * sv_position['delT_sv']
    pr = pr.to_numpy()
    sv_position['Psuedo-range'] = pr
    sv_position['CN0'] = one_epoch['Cn0DbHz']

    v_k = np.arctan2(np.sqrt(1 - ephemeris['e'].pow(2)) * sinE_k, (cosE_k - ephemeris['e']))

    Phi_k = v_k + ephemeris['omega']

    sin2Phi_k = np.sin(2 * Phi_k)
    cos2Phi_k = np.cos(2 * Phi_k)

    du_k = ephemeris['C_us'] * sin2Phi_k + ephemeris['C_uc'] * cos2Phi_k
    dr_k = ephemeris['C_rs'] * sin2Phi_k + ephemeris['C_rc'] * cos2Phi_k
    di_k = ephemeris['C_is'] * sin2Phi_k + ephemeris['C_ic'] * cos2Phi_k

    u_k = Phi_k + du_k

    r_k = A * (1 - ephemeris['e'] * np.cos(E_k)) + dr_k

    i_k = ephemeris['i_0'] + di_k + ephemeris['IDOT'] * sv_position['t_k']

    x_k_prime = r_k * np.cos(u_k)
    y_k_prime = r_k * np.sin(u_k)

    Omega_k = ephemeris['Omega_0'] + (ephemeris['OmegaDot'] - OmegaDot_e) * sv_position['t_k'] - OmegaDot_e * ephemeris[
        't_oe']

    sv_position['Sat.X'] = x_k_prime * np.cos(Omega_k) - y_k_prime * np.cos(i_k) * np.sin(Omega_k)
    sv_position['Sat.Y'] = x_k_prime * np.sin(Omega_k) + y_k_prime * np.cos(i_k) * np.cos(Omega_k)
    sv_position['Sat.Z'] = y_k_prime * np.sin(i_k)

    return sv_position


def get_positions(measurements: pd.DataFrame) -> tuple:
    manager = EphemerisManager("data")
    epoch = 0
    num_sats = 0
    while num_sats < 5:
        one_epoch = measurements.loc[
            (measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)].drop_duplicates(subset='SvName')
        timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
        one_epoch.set_index('SvName', inplace=True)
        num_sats = len(one_epoch.index)
        epoch += 1

    sats = one_epoch.index.unique().tolist()
    ephemeris = manager.get_ephemeris(timestamp, sats)

    # Run the function and check out the results:
    sv_position = calculate_satellite_position(ephemeris, timestamp, one_epoch)
    return manager, sv_position, one_epoch


def least_squares(xs, measured_pseudorange, x0, b0) -> tuple:
    dx = 100 * np.ones(3)
    # set up the G matrix with the right dimensions. We will later replace the first 3 columns
    # note that b here is the clock bias in meters equivalent, so the actual clock bias is b/LIGHTSPEED
    G = np.ones((measured_pseudorange.size, 4))
    iterations = 0
    while np.linalg.norm(dx) > 1e-3:
        # Eq. (2):
        r = np.linalg.norm(xs - x0, axis=1)
        # Eq. (1):
        phat = r + b0
        # Eq. (3):
        deltaP = measured_pseudorange - phat
        G[:, 0:3] = -(xs - x0) / r[:, None]
        # Eq. (4):
        sol = np.linalg.inv(np.transpose(G) @ G) @ np.transpose(G) @ deltaP
        # Eq. (5):
        dx = sol[0:3]
        db = sol[3]
        x0 = x0 + dx
        b0 = b0 + db
    norm_dp = np.linalg.norm(deltaP)
    return x0, b0, norm_dp


def calculate_xyz(measurements: pd.DataFrame, sv_position: pd.DataFrame, manager) -> np.ndarray:
    b0 = 0
    x0 = np.array([0, 0, 0])
    xs = sv_position[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()

    # Apply satellite clock bias to correct the measured pseudorange values
    pr = sv_position["Psuedo-range"]

    x, b, dp = least_squares(xs, pr, x0, b0)

    ecef_list = []
    for epoch in measurements['Epoch'].unique():
        one_epoch = measurements.loc[(measurements['Epoch'] == epoch) & (measurements['prSeconds'] < 0.1)]
        one_epoch = one_epoch.drop_duplicates(subset='SvName').set_index('SvName')
        if len(one_epoch.index) > 4:
            timestamp = one_epoch.iloc[0]['UnixTime'].to_pydatetime(warn=False)
            sats = one_epoch.index.unique().tolist()
            ephemeris = manager.get_ephemeris(timestamp, sats)
            sv_position = calculate_satellite_position(ephemeris, timestamp, one_epoch)
            xs = sv_position[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()
            pr = sv_position['Psuedo-range']

            x, b, dp = least_squares(xs, pr, x, b)
            ecef_list.append(x)

    # Perform coordinate transformations using the Navpy library

    ecef_array = np.stack(ecef_list, axis=0)
    return ecef_array


def calculate_Lat_Lon_Alt(ecef_array: np.ndarray) -> pd.DataFrame:
    # initial guesses of receiver clock bias and position
    if ecef_array.ndim == 1:
        Lat_Lon_Alt_array = np.stack(navpy.ecef2lla(ecef_array), axis=0).reshape(-1, 3)
    else:
        Lat_Lon_Alt_array = np.stack(navpy.ecef2lla(ecef_array), axis=1)

    # Convert back to Pandas and save to csv
    Lat_Lon_Alt_df = pd.DataFrame(Lat_Lon_Alt_array, columns=['Latitude', 'Longitude', 'Altitude'])
    return Lat_Lon_Alt_df


def get_kml(Lat_Lon_Alt_df: np.ndarray) -> simplekml.Kml:
    kml = simplekml.Kml()

    coords = []

    for row in range(len(Lat_Lon_Alt_df)):
        lat = Lat_Lon_Alt_df[row][0]
        lon = Lat_Lon_Alt_df[row][1]
        alt = Lat_Lon_Alt_df[row][2]
        coords.append((lon, lat, alt))

    for coord in coords:
        point = kml.newpoint(coords=[coord])
        point.name = f"Point at {coord[0]}, {coord[1]}"

    return kml


def main(input_filepath: str):
    """
    Main function to read data, process it, and generate output files.

    Args:
        input_filepath (str): File path to the input data.
    """
    file_name: str = input_filepath.split(".")[0]
    measurements, android_fixes = open_file(input_filepath)
    manager, sv_positions, one_epoch = get_positions(measurements)
    position_x_y_z = calculate_xyz(measurements, sv_positions, manager)
    lla_df = calculate_Lat_Lon_Alt(position_x_y_z)

    pos_xyz = np.array([0, 0, 0])
    b = 0
    xs = sv_positions[['Sat.X', 'Sat.Y', 'Sat.Z']].to_numpy()
    pr = sv_positions['Psuedo-range']
    pos_xyz, b, dp = least_squares(xs, pr, pos_xyz, b)

    sv_positions[["Pos.X", "Pos.Y", "Pos.Z"]] = pos_xyz
    sv_positions[["Lat", "Lon", "Alt"]] = calculate_Lat_Lon_Alt(np.stack(pos_xyz, axis=0)).to_numpy()[0]
    kml = get_kml(lla_df.to_numpy())

    sv_positions = sv_positions[
        ["GPS time", "Sat.X", "Sat.Y", "Sat.Z", "Psuedo-range", "CN0", "Pos.X", "Pos.Y", "Pos.Z", "Lat", "Lon", "Alt"]]
    sv_positions.to_csv(f'{file_name}-.csv')
    kml.save(f"{file_name}-.kml")


if __name__ == "__main__":
    while True:
        file_to_run = input("Enter driving/walking/fixed: ")
        if file_to_run == "driving":
            main("Driving/gnss_log_2024_04_13_19_53_33.txt")
            break
        elif file_to_run == "walking":
            main("Walking/gnss_log_2024_04_13_19_52_00.txt")
            break
        elif file_to_run == "fixed":
            main("Fixed/gnss_log_2024_04_13_19_51_17.txt")
            break
        else:
            print("wrong input, try again")
