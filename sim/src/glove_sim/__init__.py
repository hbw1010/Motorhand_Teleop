"""Standalone Motorhand glove force-feedback simulation.

This package deliberately has no ROS 2 / Wuji-hand dependency.  The real glove
backend uses 驱动 1 (``encos_hand``) or 驱动 2 (G4 gateway C++ SDK); the simulation uses MuJoCo 4.0.
"""

__version__ = "0.1.0"
