from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


CONTACT_SCHEMA_VERSION = 1
FEATURE_WIDTHS = {"active": 1, "separation": 1, "normal": 3, "position": 3}


def contact_feature_dim(features: Sequence[str], max_contacts: int) -> int:
    unknown = set(features).difference(FEATURE_WIDTHS)
    if unknown:
        raise ValueError(f"Unsupported contact features: {sorted(unknown)}")
    if max_contacts < 0:
        raise ValueError("max_contacts cannot be negative")
    return max_contacts * sum(FEATURE_WIDTHS[name] for name in features)


def encode_contacts(
    contacts: Sequence[Mapping[str, object]],
    *,
    features: Sequence[str],
    max_contacts: int,
) -> np.ndarray:
    """Encode simulator-independent contact records into a sorted padded vector."""
    width = contact_feature_dim(features, max_contacts)
    encoded = np.zeros((max_contacts, width // max_contacts if max_contacts else 0), dtype=np.float32)
    ordered = sorted(contacts, key=lambda contact: tuple(contact.get("pair", ())))[:max_contacts]
    for index, contact in enumerate(ordered):
        values = []
        for name in features:
            if name == "active":
                values.append(1.0)
            elif name == "separation":
                values.append(float(contact["separation"]))
            else:
                values.extend(np.asarray(contact[name], dtype=np.float32).reshape(-1).tolist())
        encoded[index] = values
    return encoded.reshape(-1)