# Graph Schema

## TransactionEvent

```python
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime

class TransactionEvent(BaseModel):
    tx_id: str
    timestamp: datetime
    step: int = Field(ge=0)
    src_node: int = Field(ge=0)
    dst_node: int = Field(ge=0)
    amount: float = Field(gt=0)
    normalized_amount: Optional[float] = Field(default=None, ge=0)
    currency: str = Field(min_length=3, max_length=3)
    type: Optional[str] = None
    channel: Optional[str] = None
    is_sar: Optional[int] = Field(default=None, ge=0, le=1)
```

## Constraints
- `timestamp`: ISO-8601, UTC preferred.
- `amount`: strictly positive.
- `src_node != dst_node` unless self-loop is explicitly allowed.
- `step`: integer time bucket, usually daily.
- `currency`: upper-case 3-letter code.
- `src_node` / `dst_node`: already resolved entity IDs.

## WindowGraphSnapshot

```python
from pydantic import BaseModel
from datetime import datetime

class WindowGraphSnapshot(BaseModel):
    window_start: datetime
    window_end: datetime
    nodes: list[int]
    edges: list[TransactionEvent]
```

## Output artifacts
- community assignment per node
- motif instances per window
- numeric feature table for model training/inference
