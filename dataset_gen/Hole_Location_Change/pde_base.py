class PDEBase:
    """Base class for all PDE variants."""

    def generate_U_file(self, folder, i_c, j_c, coefficients, hole_size=0.0625):
        """
        Base method. By default, raises an error indicating
        subclasses must implement it.
        """
        raise NotImplementedError("Subclasses must implement generate_U_file().")

    def generate_random_coefficients(self, p=None):
        """
        Base method. By default, raises an error indicating
        subclasses must implement it (if needed).
        """
        raise NotImplementedError("Subclasses must implement generate_random_coefficients().")