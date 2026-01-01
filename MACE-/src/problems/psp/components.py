from src.problems.base.components import BaseSolution, BaseOperator
from typing import List, Tuple, Optional, Dict
import copy
import random

class Solution(BaseSolution):
    """The solution of Port Scheduling Problem supporting multi-tugboat services.
    Attributes:
        vessel_assignments (dict): Maps vessel_id to (berth_id, start_time) tuple. None if vessel is unassigned.
        tugboat_inbound_assignments (dict): Maps vessel_id to list of tuple (tugboat_id, start_time) for inbound service.
        tugboat_outbound_assignments (dict): Maps vessel_id to list of tuple (tugboat_id, start_time) for outbound service.
    """
    def __init__(self, vessel_assignments: dict = None, 
                 tugboat_inbound_assignments: dict = None, 
                 tugboat_outbound_assignments: dict = None):
        self.vessel_assignments = vessel_assignments or {}
        self.tugboat_inbound_assignments = tugboat_inbound_assignments or {}
        self.tugboat_outbound_assignments = tugboat_outbound_assignments or {}
    
    def __str__(self) -> str:
        result = "Port Scheduling Solution:\n"
        result += "Vessel Assignments:\n"
        for vessel_id, assignment in self.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                result += f"  Vessel {vessel_id}: Berth {berth_id}, Start Time {start_time}\n"
            else:
                result += f"  Vessel {vessel_id}: Unassigned\n"
        
        result += "Inbound Tugboat Services:\n"
        for vessel_id, services in self.tugboat_inbound_assignments.items():
            if services:
                tugboat_list = ", ".join([f"Tugboat {tug_id} at time {start_time}" for tug_id, start_time in services])
                result += f"  Vessel {vessel_id}: {tugboat_list}\n"
        
        result += "Outbound Tugboat Services:\n"
        for vessel_id, services in self.tugboat_outbound_assignments.items():
            if services:
                tugboat_list = ", ".join([f"Tugboat {tug_id} at time {start_time}" for tug_id, start_time in services])
                result += f"  Vessel {vessel_id}: {tugboat_list}\n"
        
        return result

# ================== Core Assignment Operators ==================

class CompleteVesselAssignmentOperator(BaseOperator):
    """Assign a vessel to berth with complete multi-tugboat services and feasibility validation."""
    def __init__(self, vessel_id: int, berth_id: int, start_time: int, 
                 inbound_tugboats: List[Tuple[int, int]], 
                 outbound_tugboats: List[Tuple[int, int]]):
        self.vessel_id = vessel_id
        self.berth_id = berth_id
        self.start_time = start_time
        self.inbound_tugboats = inbound_tugboats
        self.outbound_tugboats = outbound_tugboats

    def run(self, solution: Solution) -> Solution:
        # Basic feasibility validation
        if not self.inbound_tugboats or not self.outbound_tugboats:
            return copy.deepcopy(solution)  # Incomplete service assignment
        
        # Validate tugboat service timing consistency (all tugboats in same service start simultaneously)
        if len(self.inbound_tugboats) > 1:
            inbound_times = [start_time for _, start_time in self.inbound_tugboats]
            if len(set(inbound_times)) > 1:
                return copy.deepcopy(solution)  # Inconsistent inbound timing
        
        if len(self.outbound_tugboats) > 1:
            outbound_times = [start_time for _, start_time in self.outbound_tugboats]
            if len(set(outbound_times)) > 1:
                return copy.deepcopy(solution)  # Inconsistent outbound timing
        
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        new_tugboat_inbound = copy.deepcopy(solution.tugboat_inbound_assignments)
        new_tugboat_outbound = copy.deepcopy(solution.tugboat_outbound_assignments)
        
        new_vessel_assignments[self.vessel_id] = (self.berth_id, self.start_time)
        new_tugboat_inbound[self.vessel_id] = copy.deepcopy(self.inbound_tugboats)
        new_tugboat_outbound[self.vessel_id] = copy.deepcopy(self.outbound_tugboats)
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )

class UnassignVesselOperator(BaseOperator):
    """Remove assignment for a vessel."""
    def __init__(self, vessel_id: int):
        self.vessel_id = vessel_id

    def run(self, solution: Solution) -> Solution:
        new_vessel_assignments = copy.deepcopy(solution.vessel_assignments)
        new_tugboat_inbound = copy.deepcopy(solution.tugboat_inbound_assignments)
        new_tugboat_outbound = copy.deepcopy(solution.tugboat_outbound_assignments)
        
        new_vessel_assignments[self.vessel_id] = None
        new_tugboat_inbound[self.vessel_id] = []
        new_tugboat_outbound[self.vessel_id] = []
        
        return Solution(
            vessel_assignments=new_vessel_assignments,
            tugboat_inbound_assignments=new_tugboat_inbound,
            tugboat_outbound_assignments=new_tugboat_outbound
        )