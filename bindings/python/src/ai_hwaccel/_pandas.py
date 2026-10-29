"""Optional pandas export. Requires pandas (the ``pandas`` extra).

    pip install pandas
"""

from __future__ import annotations

import dataclasses


def profiles_to_dataframe(profiles: list):
    try:
        import pandas as pd
    except ImportError as exc:  # pragma: no cover - exercised only without pandas
        raise ImportError(
            "pandas is required for DataFrame export; "
            "install it with: pip install pandas"
        ) from exc

    return pd.DataFrame([dataclasses.asdict(p) for p in profiles])
