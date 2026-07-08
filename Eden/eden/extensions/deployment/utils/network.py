"""Network-interface discovery helpers for Unitree deployment."""

from __future__ import annotations

import os


def _list_net_ifaces() -> list[str]:
    """Best-effort list of network interface names on Linux."""
    sys_class_net = "/sys/class/net"
    try:
        return sorted([name for name in os.listdir(sys_class_net) if name])
    except Exception:
        return []


def _iface_operstate(iface: str) -> str | None:
    try:
        with open(f"/sys/class/net/{iface}/operstate") as f:
            return f.read().strip()
    except Exception:
        return None


def pick_unitree_network_interface() -> str | None:
    """Pick a reasonable default interface for DDS traffic."""
    override = (
        os.environ.get("GS_UNITREE_NETWORK_INTERFACE")
        or os.environ.get("UNITREE_NETWORK_INTERFACE")
        or os.environ.get("CYCLONEDDS_NETWORK_INTERFACE")
    )
    if override:
        return override

    ifaces = _list_net_ifaces()
    if not ifaces:
        return None

    def is_up(name: str) -> bool:
        return _iface_operstate(name) == "up"

    def is_virtual_or_loopback(name: str) -> bool:
        return name in {"lo"} or name.startswith(("docker", "br-", "veth", "tailscale", "tun", "tap"))

    wired_prefixes = ("en", "eth")
    wired = [n for n in ifaces if n.startswith(wired_prefixes) and is_up(n) and not is_virtual_or_loopback(n)]
    if wired:
        return wired[0]

    other = [n for n in ifaces if is_up(n) and not is_virtual_or_loopback(n)]
    return other[0] if other else None
