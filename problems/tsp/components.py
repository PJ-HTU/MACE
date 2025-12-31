from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of TSP.
    A list of integers where each integer represents a node (city) in the TSP tour.
    The order of the nodes in the list defines the order in which the cities are visited in the tour.
    """
    def __init__(self, tour: list[int]):
        self.tour = tour

    def __str__(self) -> str:
        if len(self.tour) > 0:
            return "tour: " + "->".join(map(str, self.tour + [self.tour[0]]))
        return "tour: "


class AppendOperator(BaseOperator):
    """Append the node at the end of the solution."""
    def __init__(self, node: int):
        self.node = node

    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour + [self.node]
        return Solution(new_tour)


class InsertOperator(BaseOperator):
    """Insert the node into the solution at the target position."""
    def __init__(self, node: int, position: int):
        self.node = node
        self.position = position

    def run(self, solution: Solution) -> Solution:
        new_tour = solution.tour[:self.position] + [self.node] + solution.tour[self.position:]
        return Solution(new_tour)