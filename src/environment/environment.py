import time
from src.environment.LMCat_control.controller import Controller
from src.environment.LMCat_control.observer import Observer

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
            'Image', 'CH4', 
            num=num_frames, 
            scan=scan
        )
        return measurements


    def act(self, ch4_action=None):
        """
        Applies an action to the reactor.
        
        Args:
            ch4_action (float): The target CH4 flow rate.
                               
        Returns:
            dict: The new observed state after the action is applied.
        """
        # 1. Apply the action
        # If action is none, keep doing the same
        if ch4_action is None:
            ch4_action = self.observe()['CH4'][-1]

        print(f"Applying action: Setting CH4 flow to {ch4_action:.2f} sccm.")
        
        return self.controller.set_flow_CH4(ch4_action) 
    

    # Not necessary
    def set_background_gases(self, ar_flow = 200, h2_flow = 20):
        """Helper to set background gases that shouldn't change."""
        self.controller.set_flow_Ar(ar_flow)
        self.controller.set_flow_H2(h2_flow)