"""Path translation between the Immich server and this container.

Why this exists
---------------
`originalPath` from the Immich API is expressed from the **Immich server
container's** filesystem perspective, e.g.:

    /usr/src/app/upload/upload/<userId>/ab/cd/<uuid>.jpg

That path is meaningless inside *this* container. A read-only bind-mount alone
does NOT fix it: the mount changes where the bytes live, not what the API says.
The path string itself must be rewritten:

    strip IMMICH_INTERNAL_PREFIX  ->  prepend LOCAL_MOUNT

Everything here is POSIX (forward-slash) regardless of the host OS the code is
edited on, because both the Immich server and this container are Linux. We use
``posixpath`` explicitly so editing on Windows can't sneak in backslashes.
"""

from __future__ import annotations

import posixpath


def _norm(p: str) -> str:
    """Collapse redundant slashes; keep it POSIX. Does not touch '..'."""
    return posixpath.normpath(p.replace("\\", "/"))


def translate_path(original_path: str, internal_prefix: str, local_mount: str) -> str:
    """Rewrite an Immich-server path into this container's local mount path.

    Args:
        original_path: the raw ``originalPath`` from the Immich API.
        internal_prefix: the prefix the Immich server uses internally
            (IMMICH_INTERNAL_PREFIX), e.g. ``/usr/src/app/upload``.
        local_mount: where that library is mounted here (LOCAL_MOUNT),
            e.g. ``/immich-library``.

    Returns:
        The translated absolute path inside this container.

    Raises:
        ValueError: if ``original_path`` does not start with ``internal_prefix``.
            That mismatch almost always means IMMICH_INTERNAL_PREFIX is wrong —
            run the verify step (main.py prints the raw path) and adjust it.
    """
    op = _norm(original_path)
    prefix = _norm(internal_prefix)
    mount = _norm(local_mount)

    # Match on a path-segment boundary so '/upload' doesn't match '/uploads'.
    if op == prefix:
        relative = ""
    elif op.startswith(prefix.rstrip("/") + "/"):
        relative = op[len(prefix.rstrip("/")):]
    else:
        raise ValueError(
            f"originalPath {original_path!r} does not start with "
            f"IMMICH_INTERNAL_PREFIX {internal_prefix!r}. "
            f"The internal prefix is likely misconfigured — check the raw "
            f"path printed by the verify step."
        )

    relative = relative.lstrip("/")
    return posixpath.join(mount, relative)
