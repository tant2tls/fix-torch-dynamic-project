"""Package init: runs before anything else in this package is imported.

Its job is to make compilation in this repo *deterministic*, which matters
because an intermittent failure in a bug-reproduction repo is worse than no
repo: a reader cannot tell the bug from noise.

Two settings, both applied before torch is imported (torch reads them at import
time), and both overridable by the caller.

1. `TORCHINDUCTOR_COMPILE_THREADS=1`

   Inductor defaults to `compile_threads=32` here, forking parallel codegen
   workers. Triton's cache writer (`triton/runtime/cache.py:108` in 2.3.1) does:

       temp_path = f"{filepath}.tmp.pid_{pid}_{rnd_id}"
       with open(temp_path, mode) as f: f.write(data)
       os.replace(temp_path, filepath)

   The `os.replace` is atomic, but nothing holds a lock across
   create-then-rename, and the enclosing cache directory is shared by every
   worker. Under concurrency this intermittently raises

       FileNotFoundError: ... '.../triton_.cubin.tmp.pid_145798_93312'

   Measured here: about 1 run in 18 of the test suite, on both NFS and local
   /tmp -- so it is a concurrency race, not a filesystem-type problem. Serial
   compilation removes it. This repo compiles tiny graphs, so the cost is
   negligible; do NOT copy this setting into a real workload.

2. `TORCHINDUCTOR_CACHE_DIR` -> node-local disk

   Independently, Inductor defaults its cache to `$HOME/.torchinductor_cache`.
   Where `$HOME` is a shared network mount (as on many clusters) that directory
   is shared with every concurrent job on every node, which widens the same race
   and makes runs depend on unrelated jobs. A node-local cache is strictly
   better here, and the cache is a pure derived artifact.

   An explicitly-set value is honoured unless it points into a network-mounted
   `$HOME` -- exactly the bad case -- where it is overridden with a printed
   notice rather than silently.
"""

import os
import tempfile

_LOCAL_CACHE = os.path.join(tempfile.gettempdir(),
                            f"torchinductor_cache_{os.getuid()}")


def _is_network_home(path: str) -> bool:
    """True if `path` is under $HOME and $HOME is on a network filesystem.

    Conservative: only /proc/mounts evidence counts, so on a laptop with an
    ext4/apfs home this returns False and the user's setting is left alone.
    """
    try:
        home = os.path.realpath(os.path.expanduser("~"))
        real = os.path.realpath(path)
        if not (real == home or real.startswith(home + os.sep)):
            return False

        network_fs = {"nfs", "nfs4", "cifs", "smb3", "smbfs", "lustre", "gpfs",
                      "afs", "fuse.sshfs", "beegfs", "ceph", "glusterfs"}
        best_len, best_fs = -1, ""
        with open("/proc/mounts", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                parts = line.split()
                if len(parts) < 3:
                    continue
                mount_point, fstype = parts[1], parts[2]
                # Longest matching mount point wins.
                if (home == mount_point
                        or home.startswith(mount_point.rstrip(os.sep) + os.sep)):
                    if len(mount_point) > best_len:
                        best_len, best_fs = len(mount_point), fstype
        return best_fs in network_fs
    except OSError:
        return False


# (1) Serial compilation -- the actual fix for the flake.
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")

# (2) Keep the cache off a shared network home.
_configured = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
if not _configured:
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _LOCAL_CACHE
elif _is_network_home(_configured):
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _LOCAL_CACHE
    print(f"[repro] TORCHINDUCTOR_CACHE_DIR was {_configured}, which is on a "
          f"network-mounted $HOME.\n"
          f"[repro] Using node-local {_LOCAL_CACHE} instead, so runs do not "
          f"depend on other jobs.\n"
          f"[repro] Set the variable to a node-local path to silence this.")
