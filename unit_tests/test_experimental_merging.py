from artlib.experimental.merging import find, merge_objects, union


def test_find_applies_path_compression():
    parent = [0, 0, 1, 2]
    root = find(parent, 3)
    assert root == 0
    assert parent[3] == 0


def test_union_merges_sets_by_rank():
    parent = [0, 1, 2, 3]
    rank = [0, 0, 0, 0]

    union(parent, rank, 0, 1)
    union(parent, rank, 2, 3)
    union(parent, rank, 1, 3)

    assert find(parent, 0) == find(parent, 3)


def test_merge_objects_groups_connected_components():
    objects = [0, 1, 2, 3, 4]

    def can_merge(a, b):
        return abs(a - b) == 1 and min(a, b) in {0, 3}

    groups = merge_objects(objects, can_merge)
    groups_sorted = sorted(sorted(group) for group in groups)

    assert groups_sorted == [[0, 1], [2], [3, 4]]
