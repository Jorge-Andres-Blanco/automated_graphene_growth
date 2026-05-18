from blissclient import BlissClient


class Controller:
    """
    Controller interface for the LMCat reactor gas flows and pressure PIDs.

    This class uses the BlissClient to control the flow rates over Mass Flow Controllers (MFCs).

    Attributes:
        client (BlissClient): The RPC client connected to the BLISS server.
        session: The specific BLISS session context for command execution.
    
    REQUIREMENT: 
    The BLISS session on the lab computer MUST have the following functions defined:
    - set_flow_CH4(F)
    # Optional:
    - set_flow_Ar(F)
    - set_flow_H2(F)
    - set_reactor_pressure(P)

    Example:
        >>> controller = LMCatController()
        >>> # Set Argon flow to 50 sccm
        >>> response = controller.set_flowAr(50)
        >>> # Set reactor pressure to 0.2 bar
        >>> controller.set_reactor_pressure(0.2)
    """

    def __init__(self, data_store_url="redis://lid10lmcatctrl:25002", session_name="lmcat_ctrl"):

        """
        Initializes the connection to the LMCat control server.

        Args:
            data_store_url (str): The base URL for the BlissClient server.
        """

        self.client=BlissClient('http://lid10lmcatctrl:8080')
        self.session = self.client.session

    

    def set_flow_CH4(self, flow_rate):

        """
        Sets the CH4 (Methane) flow rate.

        Args:
            flow_rate (float): Target flow rate in standard units (e.g., sccm).

        Returns:
            varies: The result of the remote call, typically confirming the setpoint.
        """

        future = self.session.call("set_flow_CH4", flow_rate)
        return future.get()
    
    
    def set_flow_Ar(self, flow_rate):

        """
        Sets the Ar (Argon) flow rate.

        Args:
            flow_rate (float): Target flow rate in standard units.

        Returns:
            varies: Confirmation from the controller.
        """

        future = self.session.call("set_flow_Ar", flow_rate)
        return future.get()
    

    def set_flow_H2(self, flow_rate):

        """
        Sets the H2 (Hydrogen) flow rate.

        Args:
            flow_rate (float): Target flow rate in standard units.

        Returns:
            varies: Confirmation from the controller.
        """

        future = self.session.call("set_flow_H2", flow_rate)
        return future.get()
    

    def set_reactor_pressure(self, pressure_bar):

        """
        Sets the target pressure for the reactor PID controller.

        Args:
            pressure_bar (float): Target pressure in bar.

        Returns:
            varies: The response from the pressure regulation system.
        """

        future = self.session.call("set_reactor_pressure", pressure_bar)
        return future.get()
