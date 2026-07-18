from __future__ import annotations

import os
import warnings


os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
warnings.filterwarnings(
    "ignore",
    message=r"`torch_geometric\.distributed` has been deprecated.*",
    category=DeprecationWarning,
)
