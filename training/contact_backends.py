from __future__ import annotations

import numpy as np
import warp as wp

from training.contact import encode_contacts


def mujoco_warp_features(solver, *, num_envs: int, features, max_contacts: int) -> np.ndarray:
    """Extract contact records from a MuJoCo-Warp solver into generic features."""
    data = solver.mjw_data
    count = getattr(data, "ncon", getattr(data, "nacon", None))
    if count is None:
        raise RuntimeError("MuJoCo-Warp data does not expose a contact count")
    total = min(int(wp.to_torch(count).sum().item()), wp.to_torch(data.contact.dist).numel())
    records = [[] for _ in range(num_envs)]
    if total:
        contact = data.contact
        geometry = wp.to_torch(contact.geom).cpu().numpy()
        separation = wp.to_torch(contact.dist).cpu().numpy()
        frame = wp.to_torch(contact.frame).cpu().numpy()
        position = wp.to_torch(contact.pos).cpu().numpy()
        world_id = wp.to_torch(contact.worldid).cpu().numpy()
        for index in range(total):
            world = int(world_id[index])
            if 0 <= world < num_envs:
                records[world].append({
                    "pair": tuple(sorted((int(geometry[index, 0]), int(geometry[index, 1])))),
                    "separation": float(separation[index]),
                    "normal": frame[index, 0],
                    "position": position[index],
                })
    return np.stack([encode_contacts(items, features=features, max_contacts=max_contacts) for items in records])