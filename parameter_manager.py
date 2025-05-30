import mesh

class ParameterCoordinator:
    def __init__(self, config, mesh_iface):
        self.config = config
        self.mesh = mesh_iface

    def decide_new_params(self, metrics):
        # Decide new group parameter
        return {}

    def broadcast_change(self, new_params):
        # Broadcast new params to all nodes
        pass

    def apply_new_params(self, new_params):
        """
        Delegate parameter application to the mesh interface.
        """
        self.mesh.apply_new_params(new_params)
