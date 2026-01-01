from src.problems.base.components import BaseSolution, BaseOperator

class Solution(BaseSolution):
    """The solution of CVRP.
    A list of lists where each sublist represents a vehicle's route.
    Each sublist contains integers representing the nodes (customers) visited by the vehicle in the order of visitation.
    The routes are sorted by vehicle identifier and the nodes in the list sorted by visited order.
    """
    def __init__(self, routes: list[list[int]], depot: int):
        self.routes = routes
        self.depot = depot

    def __str__(self) -> str:
        route_string = ""
        for index, route in enumerate(self.routes):
            depot_index = route.index(self.depot)
            rotated_route = route[depot_index:] + route[:depot_index] + [self.depot]
            route = [self.depot] + route + [self.depot]
            route_string += f"vehicle_{index}: " + "->".join(map(str, rotated_route)) + "\n"
        return route_string


class AppendOperator(BaseOperator):
    """Append a node at the end of the specified vehicle's route."""
    def __init__(self, vehicle_id: int, node: int):
        self.vehicle_id = vehicle_id
        self.node = node

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        new_routes[self.vehicle_id].append(self.node)
        return Solution(new_routes, solution.depot)


class InsertOperator(BaseOperator):
    """Insert a node at a specified position within the route of a specified vehicle."""
    def __init__(self, vehicle_id: int, node: int, position: int):
        self.vehicle_id = vehicle_id
        self.node = node
        self.position = position

    def run(self, solution: Solution) -> Solution:
        new_routes = [route[:] for route in solution.routes]
        new_routes[self.vehicle_id].insert(self.position, self.node)
        return Solution(new_routes, solution.depot)