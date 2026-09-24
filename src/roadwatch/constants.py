"""Label ids and detector class groups shared by every module."""

CLASSES = ["accident", "near_miss", "red_light", "wrong_way", "illegal_u_turn",
           "stopped_vehicle", "jaywalking", "failure_to_yield", "illegal_turn",
           "solid_line_crossing", "stop_line", "congestion", "road_obstacle", "fire_smoke"]

# COCO ids -> coarse groups used by the tracker (association never crosses groups).
COCO_GROUPS = {
    0: "person",
    1: "two_wheeler",   # bicycle
    3: "two_wheeler",   # motorcycle
    2: "vehicle",       # car
    5: "vehicle",       # bus
    7: "vehicle",       # truck
    15: "animal", 16: "animal", 17: "animal", 18: "animal", 19: "animal",  # cat dog horse sheep cow
}
TRAFFIC_LIGHT = 9
COCO_NAMES = {0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
              9: "traffic light", 15: "cat", 16: "dog", 17: "horse", 18: "sheep", 19: "cow"}
DETECT_CLASSES = sorted(set(COCO_GROUPS) | {TRAFFIC_LIGHT})

ROAD_USERS = ("vehicle", "two_wheeler", "person")
MOTORISED = ("vehicle", "two_wheeler")
