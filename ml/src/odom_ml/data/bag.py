from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from rosbags.highlevel import AnyReader
from rosbags.typesys import Stores, get_types_from_msg, get_typestore

from .. import config as C

FIELDS: dict[str, tuple[str, ...]] = {
    C.TOPIC_FRONT: ("velocity",),
    C.TOPIC_REAR: ("velocity",),
    C.TOPIC_DRIVER: ("position",),
    C.TOPIC_MASTER_FIX: ("latitude", "longitude", "altitude", "status"),
    C.TOPIC_ROVER_FIX: ("latitude", "longitude", "altitude", "status"),
    C.TOPIC_MASTER_VEL: ("vx", "vy", "vz"),
    C.TOPIC_ROVER_VEL: ("vx", "vy", "vz"),
}


CUSTOM_MSGS = ("VelocitySensor", "DriverControllerCommand")


def build_typestore(msg_dir: Path | None = None) -> Stores:
    """Register the custom ``tram_vehicle_msgs`` types on a Humble typestore.

    ``Path.glob`` on a missing directory yields nothing instead of raising, so a
    wrong ``MSG_DIR`` would otherwise produce a typestore that fails much later
    with an unrelated deserialisation error.  Fail here instead.
    """
    msg_dir = Path(msg_dir) if msg_dir is not None else C.MSG_DIR
    if not msg_dir.is_dir():
        raise FileNotFoundError(
            f"message definitions not found in {msg_dir}; set ODOM_MSG_DIR to the "
            f"'msg' directory of the tram_vehicle_msgs package"
        )
    typestore = get_typestore(Stores.ROS2_HUMBLE)
    found = set()
    for path in sorted(msg_dir.glob("*.msg")):
        typename = f"tram_vehicle_msgs/msg/{path.stem}"
        typestore.register(get_types_from_msg(path.read_text(encoding="utf-8"), typename))
        found.add(path.stem)
    missing = [n for n in CUSTOM_MSGS if n not in found]
    if missing:
        raise FileNotFoundError(f"{msg_dir} is missing {missing}; found {sorted(found)}")
    return typestore


@dataclass
class RawBag:
    bag_id: str
    vehicle: str
    t0_ns: int
    t: dict[str, np.ndarray] = field(default_factory=dict)
    values: dict[str, np.ndarray] = field(default_factory=dict)
    header_stamp: dict[str, np.ndarray] = field(default_factory=dict)
    cov: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return max((float(v[-1]) for v in self.t.values() if v.size), default=0.0) / 1e9


def vehicle_of(bag_id: str) -> str:
    return bag_id.split("_")[0]


def list_bags(bags_dir: Path | None = None) -> list[str]:
    """Bag ids under the data directory.

    A bag is a directory holding a rosbag2 ``metadata.yaml``; that test keeps
    sibling directories such as ``tram_vehicle_msgs`` from being mistaken for
    one.
    """
    root = Path(bags_dir) if bags_dir is not None else C.BAGS_DIR
    if not root.is_dir():
        raise FileNotFoundError(
            f"no dataset at {root}; set ODOM_BAGS_DIR to the directory holding the bags"
        )
    return sorted(p.name for p in root.iterdir() if p.is_dir() and (p / "metadata.yaml").exists())


def _extract(msgtype: str, msg) -> tuple[float, ...]:
    if msgtype.endswith("VelocitySensor"):
        return (float(msg.velocity),)
    if msgtype.endswith("DriverControllerCommand"):
        return (float(msg.position),)
    if msgtype.endswith("NavSatFix"):
        return (
            float(msg.latitude),
            float(msg.longitude),
            float(msg.altitude),
            float(msg.status.status),
        )
    if msgtype.endswith("TwistStamped"):
        t = msg.twist.linear
        return (float(t.x), float(t.y), float(t.z))
    raise ValueError(f"unsupported msgtype {msgtype}")


def read_bag(
    bag_id: str,
    bags_dir: Path | None = None,
    topics: tuple[str, ...] = C.ALL_TOPICS,
    typestore: Stores | None = None,
) -> RawBag:
    root = Path(bags_dir) if bags_dir is not None else C.BAGS_DIR
    typestore = typestore or build_typestore()

    rows: dict[str, list[tuple[int, int, tuple[float, ...]]]] = {t: [] for t in topics}
    cov: dict[str, list[float]] = {}
    t0_ns: int | None = None

    with AnyReader([root / bag_id], default_typestore=typestore) as reader:
        conns = [c for c in reader.connections if c.topic in topics]
        for conn, timestamp, raw in reader.messages(connections=conns):
            msg = reader.deserialize(raw, conn.msgtype)
            if t0_ns is None:
                t0_ns = timestamp
            stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
            rows[conn.topic].append((timestamp, stamp_ns, _extract(conn.msgtype, msg)))
            if conn.msgtype.endswith("NavSatFix"):
                cov.setdefault(conn.topic, []).extend(float(x) for x in msg.position_covariance)

    assert t0_ns is not None, f"bag {bag_id} has no messages"
    out = RawBag(bag_id=bag_id, vehicle=vehicle_of(bag_id), t0_ns=t0_ns)
    for topic in topics:
        items = sorted(rows[topic], key=lambda r: (r[0], r[1]))
        n = len(items)
        out.t[topic] = np.fromiter((r[0] for r in items), dtype=np.int64, count=n)
        out.header_stamp[topic] = np.fromiter((r[1] for r in items), dtype=np.int64, count=n)
        width = len(FIELDS[topic])
        if n:
            out.values[topic] = np.array([r[2] for r in items], dtype=np.float64).reshape(n, width)
        else:
            out.values[topic] = np.empty((0, width), dtype=np.float64)
        if topic in cov:
            out.cov[topic] = np.asarray(cov[topic], dtype=np.float64)
    return out
