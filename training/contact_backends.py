from __future__ import annotations

import numpy as np
import torch
import warp as wp

from training.contact import encode_contacts


def mujoco_warp_features_torch(
    solver, *, num_envs: int, max_contacts: int, device: str | None = None
) -> torch.Tensor:
    """Pack MuJoCo-Warp contacts on-device without CPU/NumPy synchronization."""
    data = solver.mjw_data
    count = getattr(data, "ncon", getattr(data, "nacon", None))
    if count is None:
        raise RuntimeError("MuJoCo-Warp data does not expose a contact count")
    if max_contacts <= 0:
        target_device = device or wp.device_to_torch(solver.model.device)
        return torch.zeros((num_envs, 0), device=target_device)

    target_device = device or wp.device_to_torch(solver.model.device)
    counts = wp.to_torch(count).to(device=target_device, dtype=torch.int64)
    contact = data.contact
    distances = wp.to_torch(contact.dist).to(device=target_device)
    if distances.numel() == 0:
        return torch.zeros(
            (num_envs, max_contacts * 8), dtype=torch.float32, device=target_device
        )
    geometry = wp.to_torch(contact.geom).to(
        device=target_device, dtype=torch.int64
    )
    frames = wp.to_torch(contact.frame).to(device=target_device)
    positions = wp.to_torch(contact.pos).to(device=target_device)
    worlds = wp.to_torch(contact.worldid).to(
        device=target_device, dtype=torch.int64
    )

    contact_count = distances.shape[0]
    indices = torch.arange(contact_count, device=target_device, dtype=torch.int64)
    valid_world = (worlds >= 0) & (worlds < num_envs)
    safe_worlds = worlds.clamp(0, max(num_envs - 1, 0))
    pair_low = torch.minimum(geometry[:, 0], geometry[:, 1])
    pair_high = torch.maximum(geometry[:, 0], geometry[:, 1])
    pair_key = pair_low * (int(solver.mj_model.ngeom) + 1) + pair_high
    order_key = safe_worlds * (int(solver.mj_model.ngeom) + 1) ** 2 + pair_key
    order = torch.argsort(order_key)
    sorted_worlds = safe_worlds[order]
    sorted_indices = indices[order]
    group_start = torch.ones(contact_count, dtype=torch.bool, device=target_device)
    if contact_count > 1:
        group_start[1:] = sorted_worlds[1:] != sorted_worlds[:-1]
    group_indices = torch.where(group_start, indices, torch.zeros_like(indices))
    group_begin = torch.cummax(group_indices, dim=0).values
    slot = indices - group_begin
    count_worlds = counts.shape[0]
    slot_counts = counts[sorted_worlds.clamp(max=max(count_worlds - 1, 0))]
    active = valid_world[sorted_indices] & (slot < max_contacts) & (slot < slot_counts)

    flat_index = sorted_worlds * max_contacts + slot
    flat_index = flat_index.clamp(0, num_envs * max_contacts - 1)
    packed = torch.zeros(
        (num_envs * max_contacts, 8), dtype=torch.float32, device=target_device
    )
    selected = flat_index[active]
    selected_indices = sorted_indices[active]
    packed[selected, 0] = 1.0
    packed[selected, 1] = distances[selected_indices]
    packed[selected, 2:5] = frames[selected_indices, 0, :3]
    packed[selected, 5:8] = positions[selected_indices, :3]
    return packed.view(num_envs, max_contacts * 8)


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