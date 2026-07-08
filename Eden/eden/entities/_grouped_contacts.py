"""Pure-tensor backend for ``RigidEntity.get_grouped_contacts``.

Lives on its own so the masking / orientation / weighting logic can be
unit-tested without importing :mod:`eden.entities.rigid` (which pulls in
the support-surface mixin and triggers ``genesis.utils.array_class``
top-level code that requires ``gs.init()``).

TODO: re-home this module. Open questions for the follow-up:

* **Location.** ``eden/entities/_grouped_contacts.py`` was chosen as a
  minimal split-out from ``rigid.py``, but the contents are pure-tensor
  contact-aggregation utilities — they don't depend on the entity
  abstraction at all. ``eden/utils/contacts.py`` (next to ``pc.py``) is
  arguably the cleaner home. Decide whether contact utilities belong
  under ``eden.utils`` (the rest of the math-only helpers live there)
  or stay near the entity API as a discoverability hint.
* **Public vs private.** Filename is private (leading ``_``) but the
  exported symbols (``aggregate_grouped_contacts``,
  ``resolve_local_link_idx``) are not. Pick one: promote to public
  ``eden/utils/contacts.py`` (no underscores), or keep private and
  prefix the symbols too. ``rigid.py`` currently imports them under
  ``_`` aliases, which masks the inconsistency without resolving it.
* **Disjoint-set assumption.** ``aggregate_grouped_contacts`` assumes
  ``a_global`` and ``b_global`` describe disjoint link sets; the
  public wrapper enforces that by rejecting self-contact, but the
  helper itself does not. If the helper goes public, add an
  ``assert`` (or document the precondition more loudly) so future
  callers don't silently double-stamp cells.
* **Geom-level path.** The current ``(B, C, n_a, n_b)`` broadcast is
  fine at link-level (n_a×n_b ≤ ~30) but blows up at geom-level
  (n_a×n_b ≈ 900). Bringing geom-level on as a follow-up means
  switching to a ``scatter_add`` accumulation; the file reorg should
  happen first so the rewrite isn't done in-place under the
  underscore.
* **`resolve_local_link_idx` sharing.** Generic enough that other
  contact / link-pair APIs would benefit. Consider promoting it
  alongside the move.
"""

from __future__ import annotations

import torch


def resolve_local_link_idx(
    idx: torch.Tensor | list[int] | None,
    num_links: int,
    device: torch.device,
) -> torch.Tensor:
    """Normalise a local-link-index argument into a 1-D ``long`` tensor on ``device``.

    ``None`` is expanded to ``torch.arange(num_links)``.

    Hot-path note: when ``idx`` is already a 1-D ``long`` tensor on
    ``device``, this returns it directly with no copy or H2D transfer.
    Callers that hit the per-step path (reward / observation terms)
    should pre-resolve the indices once at ``build()`` time and pass the
    cached tensor each step — see the docstring on
    :meth:`RigidEntity.get_grouped_contacts`.
    """
    if idx is None:
        return torch.arange(num_links, device=device)
    if (
        isinstance(idx, torch.Tensor)
        and idx.dtype == torch.long
        and idx.device == device
        and idx.ndim == 1
        and idx.numel() > 0
    ):
        # Fast path: caller pre-resolved at build time. Skip dtype/device
        # checks (they are no-ops here) and the ``.tolist()`` round-trip
        # in the bounds error message — bounds are verified below.
        if (idx < 0).any() or (idx >= num_links).any():
            raise IndexError(f"link index out of range [0, {num_links}): {idx.tolist()}")
        return idx
    t = idx if isinstance(idx, torch.Tensor) else torch.as_tensor(idx)
    t = t.to(device=device, dtype=torch.long)
    if t.ndim != 1:
        raise ValueError(f"link index must be 1-D, got shape {tuple(t.shape)}")
    if t.numel() == 0:
        raise ValueError("link index cannot be empty")
    if (t < 0).any() or (t >= num_links).any():
        raise IndexError(f"link index out of range [0, {num_links}): {t.tolist()}")
    return t


def aggregate_grouped_contacts(
    *,
    link_a: torch.Tensor,
    link_b: torch.Tensor,
    position: torch.Tensor,
    force: torch.Tensor,
    valid_mask: torch.Tensor,
    a_global: torch.Tensor,
    b_global: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Force-norm-weighted per-link-pair contact aggregation.

    Inputs match the parallelised Genesis ``get_contacts`` output (env-leading,
    ``(B, C, ...)``). ``a_global`` / ``b_global`` are 1-D global link indices
    that select which links populate axes 1 and 2 of the output. Both Genesis
    orderings (a→A, b→B) and (a→B, b→A) are folded into the same pair so
    the caller never has to care which side Genesis labelled "a".

    Genesis's ``get_contacts(with_entity=other)`` does **not** canonicalize
    contact-pair ordering: its ``valid_mask`` is built as
    ``(a∈self ∨ b∈self) ∧ (a∈other ∨ b∈other)``
    (see ``RigidEntity.get_contacts`` in genesis), so both orderings reach
    this function — handling both is required, not optional.

    Returns
    -------
    dict with:

    - ``position`` : ``(B, n_a, n_b, 3)`` — force-norm-weighted mean
      contact position; zero where ``valid`` is False.
    - ``valid`` : ``(B, n_a, n_b)`` bool — True iff at least one
      contact pair contributed a **non-zero** force-norm sum
      (computed as ``force_norm_sum > 0``, matching the DexMachina
      reference). Pairs whose contacts all carry zero force — e.g. the
      first frame after make/break or quasi-static initial overlap —
      are reported invalid because no force-weighted mean position is
      defined for them. ``valid_strict_geometry = pair_mask.any(...)``
      is **not** exposed; callers needing geometric-only contact
      detection should bool-AND the raw ``get_contacts(...)["valid_mask"]``
      themselves.
    - ``force_norm_sum`` : ``(B, n_a, n_b)`` — Σ ||f_i|| over contacts
      contributing to the pair (the denominator used to weight the
      position mean). Note this is the sum of magnitudes, **not** the
      magnitude of the summed force vector — opposing-direction
      contacts do not cancel here.
    """
    la = link_a.unsqueeze(-1)  # (B, C, 1)
    lb = link_b.unsqueeze(-1)  # (B, C, 1)
    # Each match tensor below is (B, C, n_a) or (B, C, n_b); they combine into
    # the full (B, C, n_a, n_b) pair mask via outer-AND on the membership axes.
    a_in_self = (la == a_global.view(1, 1, -1)).unsqueeze(-1)  # (B, C, n_a, 1)
    b_in_other = (lb == b_global.view(1, 1, -1)).unsqueeze(-2)  # (B, C, 1,  n_b)
    a_in_other = (la == b_global.view(1, 1, -1)).unsqueeze(-2)  # (B, C, 1,  n_b)
    b_in_self = (lb == a_global.view(1, 1, -1)).unsqueeze(-1)  # (B, C, n_a, 1)
    pair_mask = (a_in_self & b_in_other) | (b_in_self & a_in_other)
    pair_mask = pair_mask & valid_mask[..., None, None]  # (B, C, n_a, n_b)

    force_norm = force.norm(dim=-1)  # (B, C)
    weight = force_norm.unsqueeze(-1).unsqueeze(-1) * pair_mask.to(force_norm.dtype)
    weighted_pos_sum = (position.unsqueeze(-2).unsqueeze(-2) * weight.unsqueeze(-1)).sum(dim=1)  # (B, n_a, n_b, 3)
    force_norm_sum = weight.sum(dim=1)  # (B, n_a, n_b)
    valid = force_norm_sum > 0
    mean_pos = torch.where(
        valid.unsqueeze(-1),
        weighted_pos_sum / force_norm_sum.unsqueeze(-1).clamp(min=1e-12),
        torch.zeros_like(weighted_pos_sum),
    )
    return {"position": mean_pos, "valid": valid, "force_norm_sum": force_norm_sum}
