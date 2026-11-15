"""Shared, monotonic deadline checks for optimization and traffic assignment."""

import time


class SolveTimeLimit(TimeoutError):
    """The overall solve deadline has expired."""


def check_deadline(deadline):
    if deadline is not None and time.monotonic() >= deadline:
        raise SolveTimeLimit("Solver time limit reached")
