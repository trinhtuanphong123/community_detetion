# ============================================================
# Cell 1: Install / Import / Mount Google Drive
# ============================================================

# Polars thường đã có sẵn trên Colab, nhưng dòng này giúp đảm bảo có bản mới đủ ổn định.
!pip -q install polars pyarrow

import os
import gc
import time
import numpy as np
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
from bisect import bisect_right

# Add to the existing import block:
from itertools import combinations, product
from dataclasses import dataclass, field    # already imported in Cell 5; add here too
                                            # so Cell 5 can stay self-contained
import polars as pl

from google.colab import drive
drive.mount("/content/drive")

print("Environment ready.")
print("Polars version:", pl.__version__)
