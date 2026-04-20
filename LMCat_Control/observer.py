from blissdata import DataStore
import numpy as np

class Observer:

    """
    Interface for interacting with the BLISS data store to retrieve sensor measurements.

    This class provides a high-level API to fetch data from specific Redis-backed 
    data stores, filtering by session and scan numbers.

    Attributes:
        data_store (DataStore): Connection instance to the BLISS Redis store.
        session_name (str): The specific BLISS session to monitor.
        sensors (dict): Mapping of human-readable names to BLISS stream keys.
    """

    
    def __init__(self, data_store_url="redis://lid10lmcatctrl:25002", session_name="lmcat_camera"):

        """
        Initializes the Observer with connection details.

        Args:
            data_store_url (str): The Redis URL for the data store.
            session_name (str): Session name used to filter scans.
        """        

        self.data_store = DataStore(data_store_url)
        
        self.session_name = session_name
        
        self.sensors = {'Image': 'basler:image',
                             'H2': 'H2:H2',
                             'Ar': 'Ar:Ar',
                             'CH4': 'CH4:CH4',
                             'Pressure': 'Pressure:Pressure',
                             'ArAux': 'ArAux:ArAux',
                             'Temperature':'nanodac_thermocouple_T:nanodac_thermocouple_T'}
    

    def get_scan(self, scan_number=-1):

        """
        Loads a specific scan object from the data store.

        Args:
            scan_number (int): Index of the scan. Use -1 for the most recent scan.
                Note: This refers to the index in the existing scans list, 
                not necessarily the physical scan ID.

        Returns:
            blissdata.scan.Scan: The loaded scan object.

        Raises:
            ValueError: If no scans are found for the configured session.
        """

        timestamp, keys = self.data_store.search_existing_scans(session=self.session_name)

        if not keys:
            raise ValueError(f"No scans found for session {self.session_name}.")
        
        if scan_number < 0:
            target_scan = self.data_store.load_scan(keys[scan_number])
        else:
            target_scan = self.data_store.load_scan(keys[scan_number-1])
            print(f'Loaded scan number {target_scan.number}\nWARNING: scan number may not correspond to the requested number (scan_number parameter is not fully implemented)')

        return target_scan
    

    def get_last_measurements(self, *measurements, num = 1, scan = -1):

        """
        Retrieves the most recent data points for given sensors.

        Args:
            *measurements (str): Variable number of sensor names to retrieve.
                Valid options: 'Image', 'H2', 'Ar', 'CH4', 'Pressure', 'ArAux', 'Temperature'.
            num (int): Number of latest data points/frames to return. Defaults to 1.
            scan (int): Scan index to query. Defaults to -1 (latest).

        Returns:
            dict: A dictionary where keys are the sensor names and values are 
                NumPy arrays containing the last 'num' data points.
                Returns None if an error occurs during retrieval.
                
        Example:
            >>> obs = Observer()
            >>> data = obs.get_last_measurements('Temperature', 'Image', num=5)
            >>> print(data['Temperature'].shape)
            (5,)
        """

        try:
            sensors = [self.sensors.get(sensor_name) for sensor_name in measurements]

        except Exception as e:
            print(f"Invalid measurement key: {e}")
            return None

        scan = self.get_scan(scan)
        

        try:
            streams = [scan.streams[s] for s in sensors]
            
            length = len(streams[0])



            if num > length:
                print(f"Requested {num} measurements, but only {length} are available. Returning all available measurements.")
                num = length

            sliced_measurements = [stream[(length-num):] for stream in streams]

            return dict(zip(measurements, sliced_measurements))
        
        except Exception as e:
            print(f"Error occurred: {e}")
            return None
        