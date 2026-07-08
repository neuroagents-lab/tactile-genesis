"""
Offline linear modal analysis for the ``ContactAudio`` sensor.

Given a tetrahedral mesh and an isotropic material, this assembles the linear-elasticity stiffness ``K`` and a lumped
mass ``M`` and solves the generalized eigenproblem ``K phi = omega^2 M phi`` to recover the object's vibration modes
(frequency, decay, and per-mode surface mode shape). The result feeds ``ContactAudio`` as physically-derived
``modal_freqs`` / ``modal_decays`` / ``modal_gains`` instead of hand-tuned values (cf. Lu & Culbertson 2023; Zheng &
James 2011).

This is a one-time precompute (numpy dense ``eigh``; intended for modest meshes -- pass a coarse ``tet_cfg`` for large
objects), not a per-step cost. Rayleigh damping ``C = alpha M + beta K`` gives each mode a decay rate
``(alpha + beta * omega^2) / 2``.
"""

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

import genesis as gs


class Material(NamedTuple):
    """
    Isotropic linear-elastic material for modal analysis.

    Parameters
    ----------
    density : float
        Mass density in kg/m^3.
    youngs_modulus : float
        Young's modulus E in Pa.
    poisson_ratio : float
        Poisson's ratio nu (dimensionless, in [0, 0.5)).
    rayleigh_alpha : float
        Mass-proportional Rayleigh damping coefficient (1/s); dominates low-frequency decay.
    rayleigh_beta : float
        Stiffness-proportional Rayleigh damping coefficient (s); dominates high-frequency decay.
    contact_damping_per_force : float
        Suggested ``ContactAudioProperties.contact_damping_per_force`` (1/(s*N)) for this material, i.e. the
        force-coupled in-contact damping (Zheng's gamma-scaled viscous contact damping). Not used by the eigensolve;
        carried here so a preset fully parameterizes the sensor.
    """

    density: float
    youngs_modulus: float
    poisson_ratio: float
    rayleigh_alpha: float
    rayleigh_beta: float
    contact_damping_per_force: float = 0.0


# Density / Young's / Poisson + Rayleigh alpha,beta from Lu & Culbertson 2023 (Table I) and Zheng & James 2011
# (Table 1); contact_damping_per_force is a sensible default derived from each material's contact-damping scale gamma.
MATERIAL_PRESETS: dict[str, Material] = {
    "abs": Material(1100.0, 2.6e9, 0.36, 4.0, 3e-7, 0.04),
    "aluminium": Material(2700.0, 6.9e10, 0.33, 0.0, 5e-7, 0.2),
    "steel": Material(7850.0, 2.0e11, 0.29, 5.0, 3e-8, 0.3),
    "wood": Material(750.0, 1.1e10, 0.25, 60.0, 4e-6, 0.05),
    "ceramic": Material(2700.0, 7.4e10, 0.19, 6.0, 1e-7, 0.03),
    "glass": Material(2500.0, 6.5e10, 0.23, 1.0, 1e-7, 0.05),
    "polystyrene": Material(1050.0, 3.5e9, 0.34, 30.0, 8e-7, 0.004),
}


@dataclass
class ModalModel:
    """
    Result of a modal analysis: the lowest non-rigid vibration modes plus their surface mode shapes.

    Attributes
    ----------
    freqs : np.ndarray
        Mode frequencies in Hz, shape ``(n_modes,)``, ascending.
    decays : np.ndarray
        Mode amplitude decay rates in 1/s, shape ``(n_modes,)`` (Rayleigh ``(alpha + beta*omega^2)/2``).
    gains : np.ndarray
        Per-mode output weight in ``[0, 1]``, shape ``(n_modes,)`` (RMS surface normal mode-shape amplitude,
        normalized). Suitable as ``ContactAudioProperties.modal_gains``.
    surface_points : np.ndarray
        World/local positions of the surface sample vertices, shape ``(n_surface, 3)``. Foundation for
        position-dependent (grasp-location) timbre.
    surface_gains : np.ndarray
        Per-surface-vertex, per-mode surface normal mode-shape amplitude, shape ``(n_surface, n_modes)``.
    """

    freqs: np.ndarray
    decays: np.ndarray
    gains: np.ndarray
    surface_points: np.ndarray
    surface_gains: np.ndarray


# 6x12 strain-displacement row layout for a constant-strain tetrahedron, indexed by (strain_row, local_node, xyz).
# Voigt strain order: [exx, eyy, ezz, gxy, gyz, gzx]. Each entry is which displacement component the node gradient
# multiplies; -1 means no contribution.
_B_ROWS = (
    (0, 0),  # exx <- dN/dx * u_x
    (1, 1),  # eyy <- dN/dy * u_y
    (2, 2),  # ezz <- dN/dz * u_z
    (3, 1),  # gxy <- dN/dy * u_x  AND  dN/dx * u_y
    (3, 0),
    (4, 2),  # gyz <- dN/dz * u_y  AND  dN/dy * u_z
    (4, 1),
    (5, 2),  # gzx <- dN/dz * u_x  AND  dN/dx * u_z
    (5, 0),
)


def _elasticity_matrix(youngs: float, poisson: float) -> np.ndarray:
    """6x6 isotropic Voigt elasticity matrix from Young's modulus and Poisson's ratio."""
    lam = youngs * poisson / ((1.0 + poisson) * (1.0 - 2.0 * poisson))
    mu = youngs / (2.0 * (1.0 + poisson))
    D = np.zeros((6, 6), dtype=np.float64)
    D[:3, :3] = lam
    D[0, 0] = D[1, 1] = D[2, 2] = lam + 2.0 * mu
    D[3, 3] = D[4, 4] = D[5, 5] = mu
    return D


def _assemble(verts: np.ndarray, elems: np.ndarray, material: Material):
    """Assemble the dense stiffness matrix K (3N x 3N) and the lumped mass diagonal (3N,)."""
    verts = np.asarray(verts, dtype=np.float64)
    elems = np.asarray(elems, dtype=np.int64)
    n = verts.shape[0]
    p = verts[elems]  # (T, 4, 3)
    # Edge matrix J (columns e1, e2, e3); volume = det(J)/6.
    J = np.stack([p[:, 1] - p[:, 0], p[:, 2] - p[:, 0], p[:, 3] - p[:, 0]], axis=2)  # (T, 3, 3)
    detJ = np.linalg.det(J)
    vol = np.abs(detJ) / 6.0
    good = vol > 1e-12
    J, p, vol = J[good], p[good], vol[good]
    elems = elems[good]
    invJ = np.linalg.inv(J)  # (T, 3, 3)

    # Shape-function gradients: rows of invJ give nodes 1,2,3; node 0 is the negative sum.
    grad = np.zeros((J.shape[0], 4, 3), dtype=np.float64)
    grad[:, 1, :] = invJ[:, 0, :]
    grad[:, 2, :] = invJ[:, 1, :]
    grad[:, 3, :] = invJ[:, 2, :]
    grad[:, 0, :] = -(grad[:, 1] + grad[:, 2] + grad[:, 3])

    # Strain-displacement matrix B (T, 6, 12).
    T = J.shape[0]
    B = np.zeros((T, 6, 12), dtype=np.float64)
    for node in range(4):
        bx, by, bz = grad[:, node, 0], grad[:, node, 1], grad[:, node, 2]
        comp = (bx, by, bz)
        col = node * 3
        B[:, 0, col + 0] = bx
        B[:, 1, col + 1] = by
        B[:, 2, col + 2] = bz
        B[:, 3, col + 0], B[:, 3, col + 1] = by, bx
        B[:, 4, col + 1], B[:, 4, col + 2] = bz, by
        B[:, 5, col + 0], B[:, 5, col + 2] = bz, bx
        del comp

    D = _elasticity_matrix(material.youngs_modulus, material.poisson_ratio)
    Ke = vol[:, None, None] * np.einsum("tki,kl,tlj->tij", B, D, B)  # (T, 12, 12)

    # Scatter into the global dense stiffness matrix.
    dof = (elems[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(T, 12)  # (T, 12)
    K = np.zeros((3 * n, 3 * n), dtype=np.float64)
    idx_i = np.broadcast_to(dof[:, :, None], (T, 12, 12))
    idx_j = np.broadcast_to(dof[:, None, :], (T, 12, 12))
    np.add.at(K, (idx_i.ravel(), idx_j.ravel()), Ke.ravel())

    # Lumped mass: each tet contributes rho*V/4 to each of its 4 nodes.
    m = np.zeros(n, dtype=np.float64)
    np.add.at(m, elems.ravel(), np.repeat(material.density * vol / 4.0, 4))
    m_dof = np.repeat(m, 3)
    return K, m_dof, elems


def _surface_normals(verts: np.ndarray, faces: np.ndarray):
    """Area-weighted per-vertex normals (only surface vertices are nonzero) and the surface vertex indices."""
    verts = np.asarray(verts, dtype=np.float64)
    fp = verts[faces]
    fn = np.cross(fp[:, 1] - fp[:, 0], fp[:, 2] - fp[:, 0])  # area-weighted face normals
    vn = np.zeros_like(verts)
    for c in range(3):
        np.add.at(vn, faces[:, c], fn)
    norm = np.linalg.norm(vn, axis=1, keepdims=True)
    surf_v = np.unique(faces)
    vn = np.where(norm > 1e-12, vn / np.maximum(norm, 1e-12), 0.0)
    return vn, surf_v


def compute_modal_model(
    verts: np.ndarray,
    elems: np.ndarray,
    material: Material,
    n_modes: int,
    sample_rate: float | None = None,
    max_vertices: int = 1200,
) -> ModalModel:
    """
    Compute the lowest ``n_modes`` non-rigid vibration modes of a tetrahedral mesh.

    Parameters
    ----------
    verts : array-like, shape (N, 3)
        Tetrahedral mesh vertices (meters).
    elems : array-like, shape (T, 4)
        Tetrahedra (vertex indices).
    material : Material
        Isotropic material (use one of ``MATERIAL_PRESETS``).
    n_modes : int
        Number of modes to return (the 6 rigid-body modes are skipped).
    sample_rate : float, optional
        If given, modes at or above ``0.45 * sample_rate / 2`` (the carrier band edge) are dropped so the sensor
        never has to synthesize an aliasing mode.
    max_vertices : int
        Guard against an accidental dense eigensolve on a huge mesh; raises above this. Pass a coarser ``tet_cfg``
        to the tetrahedralizer for large objects.
    """
    verts = np.asarray(verts, dtype=np.float64)
    if verts.shape[0] > max_vertices:
        gs.raise_exception(
            f"compute_modal_model: mesh has {verts.shape[0]} vertices > max_vertices={max_vertices}. Dense modal "
            "analysis would be slow; tetrahedralize more coarsely or raise max_vertices."
        )

    K, m_dof, elems = _assemble(verts, elems, material)

    # Mass-normalize to a symmetric standard eigenproblem: A = M^-1/2 K M^-1/2.
    minv_sqrt = 1.0 / np.sqrt(m_dof)
    A = minv_sqrt[:, None] * K * minv_sqrt[None, :]
    A = 0.5 * (A + A.T)
    w, vecs = np.linalg.eigh(A)  # ascending eigenvalues = omega^2
    phi = minv_sqrt[:, None] * vecs  # back to physical displacement modes

    # Drop the 6 rigid-body (near-zero) modes, keep positive-frequency modes (eigenvalues are ascending).
    omega2 = np.clip(w, 0.0, None)
    nonrigid = np.where(w > 1e-3 * max(w.max(), 1.0))[0]
    nonrigid = nonrigid[nonrigid >= 6] if nonrigid.size > 6 else nonrigid
    if nonrigid.size == 0:
        gs.raise_exception("compute_modal_model: no non-rigid vibration modes found (degenerate mesh?).")

    if sample_rate is not None:
        band_edge = 0.45 * (sample_rate / 2.0)
        below = nonrigid[np.sqrt(omega2[nonrigid]) / (2.0 * np.pi) < band_edge]
        if below.size == 0:
            gs.logger.warning(
                "compute_modal_model: no modes below the carrier band edge; keeping the lowest modes (the sensor's "
                "own Nyquist guard will silence any above Nyquist). Increase audio_substeps for a higher sample rate."
            )
            keep = nonrigid[:n_modes]
        else:
            keep = below[:n_modes]
    else:
        keep = nonrigid[:n_modes]

    freqs = np.sqrt(omega2[keep]) / (2.0 * np.pi)
    decays = 0.5 * (material.rayleigh_alpha + material.rayleigh_beta * omega2[keep])

    # Surface mode shapes: normal component of each mode's displacement at surface vertices.
    import igl

    faces = igl.boundary_facets(np.asarray(elems, dtype=np.int64))
    faces = faces[0] if isinstance(faces, tuple) else faces
    vn, surf_v = _surface_normals(verts, faces)
    n = verts.shape[0]
    surface_gains = np.zeros((surf_v.shape[0], keep.shape[0]), dtype=np.float64)
    for j, mode in enumerate(keep):
        disp = phi[:, mode].reshape(n, 3)
        surface_gains[:, j] = np.sum(disp[surf_v] * vn[surf_v], axis=1)

    rms = np.sqrt(np.mean(surface_gains**2, axis=0)) if surf_v.size else np.ones(keep.shape[0])
    gains = rms / max(rms.max(), 1e-12)

    return ModalModel(
        freqs=freqs.astype(np.float32),
        decays=decays.astype(np.float32),
        gains=gains.astype(np.float32),
        surface_points=verts[surf_v].astype(np.float32),
        surface_gains=surface_gains.astype(np.float32),
    )


def tetrahedralize(mesh, tet_cfg: dict | None = None):
    """
    Tetrahedralize a surface mesh (a ``trimesh.Trimesh``) into ``(verts, elems)`` using Genesis's tetgen wrapper.
    """
    import genesis.utils.mesh as mu

    return mu.tetrahedralize_mesh(mesh, tet_cfg or {})
