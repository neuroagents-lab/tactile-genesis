"""Eden extensions: optional pluggable subsystems (retargeting, sysid, ...).

Submodules are imported lazily so optional extras (e.g. ``nlopt`` for
``retargeting.dex_retargeter``) only become required when actually used.
"""
