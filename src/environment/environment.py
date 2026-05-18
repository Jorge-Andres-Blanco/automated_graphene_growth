import time
from src.environment.LMCat_control import Controller
from src.environment.LMCat_control import Observer

class ReactorEnv:
    """
    Unified Environment interface for the LMCat Graphene Reactor.
    Handles both sending actions to the hardware and observing the results.
    """
    def __init__(self, data_store_url="redis://lid10lmcatctrl:25002"):
        print("Initializing Reactor Environment...")
        self.controller = Controller(data_store_url=data_store_url)
        self.observer = Observer(data_store_url=data_store_url)
        print("Environment Ready.")

    def observe(self, num_frames=1, scan=-1):
        """
        Fetches the current state of the reactor.
        
        Args:
            num_frames (int): Number of recent data points to fetch.
            scan (int): Scan index to query (-1 for latest).
            
        Returns:
            dict: Current state containing at least 'Image' and 'CH4' flow.
        """
        # Fetch the most critical measurements for the World Model
        measurements = self.observer.get_last_measurements(
            'Image', 'CH4', 'Temperature', 'Pressure', 
            num=num_frames, 
            scan=scan
        )
        return measurements

    def step(self, ch4_action, wait_time=2.0):
        """
        Applies an action to the reactor and returns the new observed state.
        
        Args:
            ch4_action (float): The target CH4 flow rate.
            wait_time (float): Time in seconds to wait for the system to stabilize 
                               before taking the next observation.
                               
        Returns:
            dict: The new observed state after the action is applied.
        """
        # 1. Apply the action
        print(f"Applying action: Setting CH4 flow to {ch4_action:.2f} sccm")
        self.controller.set_flowCH4(ch4_action)
        
        # 2. Wait for the physical system to respond (gas travel time, sensor delay)
        # Note: In a real-time system, this wait_time should match your model's step_size interval
        time.sleep(wait_time)
        
        # 3. Observe the new state
        new_state = self.observe(num_frames=1)
        
        return new_state

    # --- Additional specific control methods if needed ---
    def set_background_gases(self, ar_flow, h2_flow):
        """Helper to set background gases that don't change every step."""
        self.controller.set_flowAr(ar_flow)
        self.controller.set_flowH2(h2_flow)