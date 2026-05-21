"""Class-id remap LUTs for SkyScapes benchmarks.

The dataset's native labels are 31 fine-grained classes (Section 2.1 of the
paper). Each benchmark merges them differently:

- Dense-20: lane markings (19-30) merged into a single class (19).
- Lane-13: non-lane (0-18) merged into a single class (0); 12 lane types kept (1-12).
- Category-11: classes grouped into 10 semantic categories + lane (10).
"""

# Source ids: 0-18 = non-lane classes; 19-30 = 12 lane-marking sub-types.

# --- Dense-20 ---
DENSE_20_MAP = {i: i for i in range(19)}
for i in range(19, 31):
    DENSE_20_MAP[i] = 19  # all lane markings → 19

# --- Lane-13 ---
# Class 0 = "non-lane" (merge of 0-18); classes 1-12 = the 12 lane types.
LANE_13_MAP = {i: 0 for i in range(19)}
for i in range(19, 31):
    LANE_13_MAP[i] = i - 18  # 19 → 1, 20 → 2, ..., 30 → 12

# --- Category-11 ---
# Categories: vegetation (0), road (1), parking (2), bike/sidewalk (3),
# entrance/danger (4), building (5), vehicle (6), clutter (7),
# impervious (8), tree (9), lane (10).
CATEGORY_11_MAP = {
    0: 0,   # low_vegetation → vegetation
    1: 1, 2: 1,  # paved_road, non_paved_road → road
    3: 2, 4: 2,  # paved_parking, non_paved_parking → parking
    5: 3, 6: 3,  # bike_way, sidewalk → bike/sidewalk
    7: 4, 8: 4,  # entrance_exit, danger_area → entrance/danger
    9: 5,   # building
    10: 6, 11: 6, 12: 6, 13: 6, 14: 6, 15: 6,  # car/trailer/van/truck/large_truck/bus → vehicle
    16: 7,  # clutter
    17: 8,  # impervious_surface
    18: 9,  # tree
}
for i in range(19, 31):
    CATEGORY_11_MAP[i] = 10  # all lane markings → lane
