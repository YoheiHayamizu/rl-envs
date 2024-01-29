import numpy as np


def generate_random_state(n):
    """ Generate a random state of block size n.

    Args:
        n (_type_): the number of blocks

    Returns:
        _type_: state
    """
    state = [1 for _ in range(n)]
    while not validate_state(state):
        state = list(np.random.randint(low=0, high=n + 1, size=n))
    return state


def validate_state(s):
    state = [0] + list(s)
    adj_list = {i: set() for i in range(len(s) + 1)}
    cardinality = 0
    for i, k in enumerate(state):
        if i == k:
            continue
        adj_list[k].add(i)
        adj_list[i].add(k)
        cardinality += 1

    # Check if the cardinality
    if cardinality != len(state) - 1:
        # print(adj_list, cardinality, len(state) - 1)
        # print("Cardinality is not correct")
        return False

    # Check if non-root node has more than one child
    for k, v in adj_list.items():
        if len(v) == 0:
            # print("Root node has no child")
            return False
        if k != 0 and len(v) > 2:
            # print("Non-root node has more than two children")
            return False

    # Check if the graph is connected
    visited = set()
    stack = [0]
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        stack.extend(adj_list[node])
    if len(visited) != len(state):
        # print("The graph is not connected")
        return False

    # print(adj_list, cardinality)

    return True


if __name__ == "__main__":
    validate_state([3, 1, 0])
    validate_state([0, 0, 0])
    validate_state([2, 3, 0])
    validate_state([3, 2, 0])
    validate_state([2, 3, 1])
    validate_state([1, 3, 0])
    validate_state([2, 3, 1, 0, 0, 0])
    validate_state([2, 4, 1, 0, 0, 0, 0, 1])
    validate_state([3, 4, 0, 0, 0, 0, 0, 1])
