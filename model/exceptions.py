# type: ignore
"""
Custom exceptions for the tensor network training framework.
"""


class SingularMatrixError(Exception):
    """
    Raised when a singular matrix is encountered during NTN optimization.

    This is the ONLY error that should be treated as a "completed" run
    (with singular=True flag) rather than a failure. The model saves
    the best state achieved before the singular matrix was encountered.

    All other errors should stop the script immediately.
    """

    def __init__(
        self, message: str = "Singular matrix encountered during optimization", epoch: int = None
    ):
        self.epoch = epoch
        if epoch is not None:
            message = f"{message} at epoch {epoch}"
        super().__init__(message)
