from blissclient import BlissClient


class LMCatController:
    """
    Controller interface for the LMCat reactor gas flows and pressure PIDs.

    This class uses the BlissClient to perform Remote Procedure Calls (RPC) 
    to a BLISS session, allowing control over Mass Flow Controllers (MFCs) 
    and pressure regulators.

    Attributes:
        client (BlissClient): The RPC client connected to the BLISS server.
        session: The specific BLISS session context for command execution.

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

    

    def set_flowCH4(self, flow_rate):

        """
        Sets the CH4 (Methane) flow rate.

        Args:
            flow_rate (float): Target flow rate in standard units (e.g., sccm).

        Returns:
            varies: The result of the remote call, typically confirming the setpoint.
        """

        future = self.session.call("set_flowCH4", flow_rate)
        return future.get()
    
    
    def set_flowAr(self, flow_rate):

        """
        Sets the Ar (Argon) flow rate.

        Args:
            flow_rate (float): Target flow rate in standard units.

        Returns:
            varies: Confirmation from the controller.
        """

        future = self.session.call("set_flowAr", flow_rate)
        return future.get()
    

    def set_flowH2(self, flow_rate):

        """
        Sets the H2 (Hydrogen) flow rate.

        Args:
            flow_rate (float): Target flow rate in standard units.

        Returns:
            varies: Confirmation from the controller.
        """

        future = self.session.call("set_flowH2", flow_rate)
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
