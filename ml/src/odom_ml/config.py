import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]

# The dataset is vendored in the repository, so nothing here depends on a path
# outside it.  The environment overrides exist so a copy kept elsewhere (a bigger
# disk, a network share, the jury's own layout) can be used without editing code.
DATA_DIR = Path(os.environ.get("ODOM_DATA_DIR", REPO_ROOT / "data"))
BAGS_DIR = Path(os.environ.get("ODOM_BAGS_DIR", DATA_DIR))
MSG_DIR = Path(os.environ.get("ODOM_MSG_DIR", DATA_DIR / "tram_vehicle_msgs" / "msg"))
PATHGRAPH_DIR = Path(os.environ.get("ODOM_PATHGRAPH_DIR", DATA_DIR / "pathgraph"))
DOCS_DIR = Path(os.environ.get("ODOM_DOCS_DIR", REPO_ROOT / "docs"))

ARTIFACTS_DIR = Path(os.environ.get("ODOM_ARTIFACTS_DIR", REPO_ROOT / "artifacts"))
CACHE_DIR = Path(os.environ.get("ODOM_CACHE_DIR", REPO_ROOT / ".cache"))

WORK_DIR = Path(os.environ.get("ODOM_WORK_DIR", REPO_ROOT / "work"))

TOPIC_FRONT = "/vehicle/front_bogie_velocity"
TOPIC_REAR = "/vehicle/rear_bogie_velocity"
TOPIC_DRIVER = "/vehicle/driver_position_cmd"
TOPIC_MASTER_FIX = "/sensing/gnss/master/fix"
TOPIC_MASTER_VEL = "/sensing/gnss/master/vel"
TOPIC_ROVER_FIX = "/sensing/gnss/rover/fix"
TOPIC_ROVER_VEL = "/sensing/gnss/rover/vel"

VEHICLE_TOPICS = (TOPIC_FRONT, TOPIC_REAR, TOPIC_DRIVER)
GNSS_TOPICS = (TOPIC_MASTER_FIX, TOPIC_MASTER_VEL, TOPIC_ROVER_FIX, TOPIC_ROVER_VEL)
ALL_TOPICS = VEHICLE_TOPICS + GNSS_TOPICS

VEHICLE_IDS = ("30618", "30639")

WHEELBASE_M = 7.55
ANTENNA_MASTER_X = -9.873
ANTENNA_ROVER_X = 2.563
ANTENNA_Z = 3.0
ANTENNA_BASELINE_M = ANTENNA_ROVER_X - ANTENNA_MASTER_X

KPH_TO_MS = 1.0 / 3.6
