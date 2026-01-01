import os
import numpy as np
import ast
from itertools import combinations
from src.problems.base.env import BaseEnv
from src.problems.psp.components import Solution


class Env(BaseEnv):
    """Port Scheduling env that stores the instance data, current solution, and problem state to support algorithm."""
    def __init__(self, data_name: str, **kwargs):
        super().__init__(data_name, "psp")
        self.construction_steps = self.instance_data["vessel_num"]
        self.key_item = "total_scheduling_cost"
        self.compare = lambda x, y: y - x  # Lower cost is better

    @property
    def is_complete_solution(self) -> bool:
        # A solution is complete when all vessels have been considered (assigned or unassigned)
        assigned_vessels = set(self.current_solution.vessel_assignments.keys())
        total_vessels = set(range(self.instance_data["vessel_num"]))
        return assigned_vessels == total_vessels

    def load_data(self, data_path: str) -> dict:
        """Load port scheduling problem instance data from text file."""
        instance_data = {}
        
        with open(data_path, 'r') as file:
            for line in file:
                line = line.strip()
                # Skip comments and empty lines
                if line.startswith('#') or not line:
                    continue
                
                # Parse parameter = value format
                if '=' in line:
                    param_name, param_value = line.split('=', 1)
                    param_name = param_name.strip()
                    param_value = param_value.strip()
                    
                    # Try to evaluate the value
                    try:
                        # Handle list format
                        if param_value.startswith('[') and param_value.endswith(']'):
                            value = ast.literal_eval(param_value)
                            instance_data[param_name] = np.array(value)
                        else:
                            # Handle single values
                            try:
                                # Try to parse as float first
                                value = float(param_value)
                                # Convert to int if it's a whole number
                                if value == int(value):
                                    value = int(value)
                                instance_data[param_name] = value
                            except ValueError:
                                # Keep as string if can't convert to number
                                instance_data[param_name] = param_value
                    except (ValueError, SyntaxError):
                        print(f"Warning: Could not parse value for {param_name}: {param_value}")
                        instance_data[param_name] = param_value
        
        return instance_data

    def init_solution(self) -> Solution:
        """Initialize an empty solution."""
        vessel_assignments = {i: None for i in range(self.instance_data["vessel_num"])}
        tugboat_inbound_assignments = {i: [] for i in range(self.instance_data["vessel_num"])}
        tugboat_outbound_assignments = {i: [] for i in range(self.instance_data["vessel_num"])}
        
        return Solution(
            vessel_assignments=vessel_assignments,
            tugboat_inbound_assignments=tugboat_inbound_assignments,
            tugboat_outbound_assignments=tugboat_outbound_assignments
        )

    def get_key_value(self, solution: Solution = None) -> float:
        """Calculate the total scheduling cost of the solution based on the mathematical model."""
        if solution is None:
            solution = self.current_solution
        
        total_cost = 0.0
        
        # Z1: Unserved vessel penalty
        # Z1 = Σᵢ M·αᵢ·(1 - Σⱼ Σₜ xᵢⱼₜ)
        unserved_penalty = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            if solution.vessel_assignments.get(vessel_id) is None:
                unserved_penalty += (self.instance_data["penalty_parameter"] * 
                                   self.instance_data["vessel_priority_weights"][vessel_id])
        
        # Z2: Total port time cost
        # Z2 = Σᵢ αᵢ·βᵢ·[Σₜ (t + τ^out_i)·z^out_it - Σₜ t·z^in_it]
        port_time_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            if assignment is not None:
                inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
                outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
                
                if inbound_services and outbound_services:
                    # For all tugboats serving the same vessel must start at the same time
                    # So we can get the service start time from any tugboat (they're all the same)
                    inbound_start = inbound_services[0][1]  # z^in_it start time
                    outbound_start = outbound_services[0][1]  # z^out_it start time
                    
                    # Calculate Z2 according to the mathematical formula
                    # Σₜ (t + τ^out_i)·z^out_it - for this vessel, only one time period is active
                    outbound_duration = self.instance_data["vessel_outbound_service_times"][vessel_id]
                    outbound_term = outbound_start + outbound_duration
                    
                    # Σₜ t·z^in_it - for this vessel, only one time period is active  
                    inbound_term = inbound_start
                    
                    port_time_cost += (self.instance_data["vessel_priority_weights"][vessel_id] * 
                                     self.instance_data["vessel_waiting_costs"][vessel_id] * 
                                     (outbound_term - inbound_term))
        
        # Z3: ETA deviation cost
        # Z3 = Σᵢ αᵢ·γᵢ·(u^early_i + u^late_i)
        # Based on constraint (16): Σₜ t·z^in_it = ETAᵢ + u^late_i - u^early_i
        eta_deviation_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            if inbound_services:
                # Get inbound service start time (z^in_it active time)
                inbound_start = inbound_services[0][1]
                eta = self.instance_data["vessel_etas"][vessel_id]
                
                # Calculate linearized ETA deviations based on constraint (16)
                # inbound_start = ETA + u^late - u^early
                if inbound_start >= eta:
                    # Late arrival case: u^early = 0, u^late = inbound_start - ETA
                    u_early = 0
                    u_late = inbound_start - eta
                else:
                    # Early arrival case: u^late = 0, u^early = ETA - inbound_start
                    u_early = eta - inbound_start  
                    u_late = 0
                
                eta_deviation_cost += (self.instance_data["vessel_priority_weights"][vessel_id] * 
                                     self.instance_data["vessel_jit_costs"][vessel_id] * 
                                     (u_early + u_late))
        
        # Z4: Tugboat utilization cost
        # Z4 = Σₖ Σᵢ Σₜ cₖ·(τ^in_i·y^in_ikt + τ^out_i·y^out_ikt)
        tugboat_cost = 0.0
        for vessel_id in range(self.instance_data["vessel_num"]):
            # Inbound tugboat services - support multiple tugboats per service
            for tugboat_id, start_time in solution.tugboat_inbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_inbound_service_times"][vessel_id]
                tugboat_unit_cost = self.instance_data["tugboat_costs"][tugboat_id]
                tugboat_cost += tugboat_unit_cost * service_duration
            
            # Outbound tugboat services - support multiple tugboats per service  
            for tugboat_id, start_time in solution.tugboat_outbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_outbound_service_times"][vessel_id]
                tugboat_unit_cost = self.instance_data["tugboat_costs"][tugboat_id]
                tugboat_cost += tugboat_unit_cost * service_duration
        
        # Total weighted cost: Z = λ₁·Z₁ + λ₂·Z₂ + λ₃·Z₃ + λ₄·Z₄
        total_cost = (self.instance_data["objective_weights"][0] * unserved_penalty +
                     self.instance_data["objective_weights"][1] * port_time_cost +
                     self.instance_data["objective_weights"][2] * eta_deviation_cost +
                     self.instance_data["objective_weights"][3] * tugboat_cost)
        
        return total_cost

    def validation_solution(self, solution: Solution = None) -> bool:
        """
        Check the validation of the solution following the mathematical model constraint
        """
        if solution is None:
            solution = self.current_solution
    
        # Basic solution format validation
        if not isinstance(solution, Solution):
            return False
    
        # Constraint (2): Vessel-Berth Compatibility Constraints  
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                vessel_size = self.instance_data["vessel_sizes"][vessel_id]
                berth_capacity = self.instance_data["berth_capacities"][berth_id]
                if berth_capacity < vessel_size:
                    return False
    
        # Constraint (3): Inbound Tugboat Service Coupling
        # Constraint (4): Outbound Tugboat Service Coupling
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
            
            if assignment is not None:
                # Assigned vessels must have both inbound and outbound services
                if not inbound_services or not outbound_services:
                    return False
            else:
                # Unassigned vessels should not have any services
                if inbound_services or outbound_services:
                    return False
    
        # Constraint (5): Tugboat Horsepower Constraints for Inbound
        # Constraint (6): Tugboat Horsepower Constraints for Outbound
        for vessel_id in range(self.instance_data["vessel_num"]):
            required_hp = self.instance_data["vessel_horsepower_requirements"][vessel_id]
            
            # Check inbound services
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            if inbound_services:
                total_hp = sum(self.instance_data["tugboat_horsepower"][tug_id] for tug_id, _ in inbound_services)
                if total_hp < required_hp:
                    return False
            
            # Check outbound services
            outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
            if outbound_services:
                total_hp = sum(self.instance_data["tugboat_horsepower"][tug_id] for tug_id, _ in outbound_services)
                if total_hp < required_hp:
                    return False
    
        # Constraint (7): Tugboat Quantity Limits for Inbound
        # Constraint (8): Tugboat Quantity Limits for Outbound
        max_tugboats = self.instance_data["max_tugboats_per_service"]
        for vessel_id in range(self.instance_data["vessel_num"]):
            inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
            outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
            
            if len(inbound_services) > max_tugboats or len(outbound_services) > max_tugboats:
                return False
    
        # Constraint (11): Berth Capacity Constraints
        berth_schedules = {}
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                berth_id, start_time = assignment
                duration = self.instance_data["vessel_durations"][vessel_id]
                
                if berth_id not in berth_schedules:
                    berth_schedules[berth_id] = []
                
                # Check for time overlapping with existing assignments
                for existing_start, existing_end in berth_schedules[berth_id]:
                    if not (start_time + duration <= existing_start or start_time >= existing_end):
                        return False
                
                berth_schedules[berth_id].append((start_time, start_time + duration))
    
        # Constraint (12): Tugboat Capacity Constraints (with Preparation Time)
        tugboat_schedules = {k: [] for k in range(self.instance_data["tugboat_num"])}
        
        for vessel_id in range(self.instance_data["vessel_num"]):
            # Process inbound services
            for tugboat_id, start_time in solution.tugboat_inbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_inbound_service_times"][vessel_id]
                prep_time = self.instance_data["inbound_preparation_time"]
                end_time = start_time + service_duration + prep_time
                
                # Check for conflicts with existing tugboat schedule
                for existing_start, existing_end in tugboat_schedules[tugboat_id]:
                    if not (end_time <= existing_start or start_time >= existing_end):
                        return False
                
                tugboat_schedules[tugboat_id].append((start_time, end_time))
            
            # Process outbound services
            for tugboat_id, start_time in solution.tugboat_outbound_assignments.get(vessel_id, []):
                service_duration = self.instance_data["vessel_outbound_service_times"][vessel_id]
                prep_time = self.instance_data["outbound_preparation_time"]
                end_time = start_time + service_duration + prep_time
                
                # Check for conflicts with existing tugboat schedule
                for existing_start, existing_end in tugboat_schedules[tugboat_id]:
                    if not (end_time <= existing_start or start_time >= existing_end):
                        return False
                
                tugboat_schedules[tugboat_id].append((start_time, end_time))
    
        # Constraint (13): Inbound Service Time Window Constraints
        for vessel_id in range(self.instance_data["vessel_num"]):
            eta = self.instance_data["vessel_etas"][vessel_id]
            early_limit = self.instance_data["vessel_early_limits"][vessel_id]
            late_limit = self.instance_data["vessel_late_limits"][vessel_id]
            
            # Check inbound service time windows
            for tugboat_id, service_start in solution.tugboat_inbound_assignments.get(vessel_id, []):
                if not (eta - early_limit <= service_start <= eta + late_limit):
                    return False
    
        # Constraint (14): Inbound-Berthing Timing Sequence Constraints
        # Constraint (15): Berthing-Outbound Timing Sequence Constraints
        for vessel_id in range(self.instance_data["vessel_num"]):
            assignment = solution.vessel_assignments.get(vessel_id)
            if assignment is not None:
                _, berth_start = assignment
                berth_duration = self.instance_data["vessel_durations"][vessel_id]
                berth_end = berth_start + berth_duration
                
                # Check inbound → berthing sequence
                inbound_services = solution.tugboat_inbound_assignments.get(vessel_id, [])
                if inbound_services:
                    inbound_start = inbound_services[0][1]  # All tugboats start at same time
                    inbound_duration = self.instance_data["vessel_inbound_service_times"][vessel_id]
                    inbound_end = inbound_start + inbound_duration
                    
                    # Berthing must start after inbound service completion
                    if berth_start < inbound_end:
                        return False
                    
                    # Check time tolerance
                    time_gap = berth_start - inbound_end
                    if time_gap > self.instance_data["time_constraint_tolerance"]:
                        return False
                
                # Check berthing to outbound sequence
                outbound_services = solution.tugboat_outbound_assignments.get(vessel_id, [])
                if outbound_services:
                    outbound_start = outbound_services[0][1]  # All tugboats start at same time
                    
                    # Outbound service must start after berth ends
                    if outbound_start < berth_end:
                        return False
                    
                    # Check time tolerance
                    time_gap = outbound_start - berth_end
                    if time_gap > self.instance_data["time_constraint_tolerance"]:
                        return False
    
        return True
        
    def get_unassigned_vessels(self, solution: Solution = None) -> list:
        """Get list of unassigned vessel IDs."""
        if solution is None:
            solution = self.current_solution
            
        unassigned = []
        for vessel_id in range(self.instance_data['vessel_num']):
            if solution.vessel_assignments.get(vessel_id) is None:
                unassigned.append(vessel_id)
        
        return unassigned

    def get_vessel_time_window(self, vessel_id: int) -> tuple:
        """Get the feasible time window for vessel's inbound service start."""
        eta = self.instance_data['vessel_etas'][vessel_id]
        early_limit = self.instance_data['vessel_early_limits'][vessel_id]
        late_limit = self.instance_data['vessel_late_limits'][vessel_id]
        
        earliest_start = max(0, int(eta - early_limit))
        latest_start = min(self.instance_data['time_periods'], int(eta + late_limit))
        
        return earliest_start, latest_start

    def get_compatible_berths(self, vessel_id: int) -> list:
        """Get list of berths compatible with vessel size requirements."""
        vessel_size = self.instance_data['vessel_sizes'][vessel_id]
        compatible_berths = []
        
        for berth_id in range(self.instance_data['berth_num']):
            if self.instance_data['berth_capacities'][berth_id] >= vessel_size:
                compatible_berths.append(berth_id)
        
        return compatible_berths

    def is_berth_available(self, berth_id: int, start_time: int, duration: int, solution: Solution = None) -> bool:
        """Check if berth is available for the given time period."""
        if solution is None:
            solution = self.current_solution
            
        end_time = start_time + duration
        
        for vessel_id, assignment in solution.vessel_assignments.items():
            if assignment is not None:
                assigned_berth, assigned_start = assignment
                if assigned_berth == berth_id:
                    assigned_duration = self.instance_data['vessel_durations'][vessel_id]
                    assigned_end = assigned_start + assigned_duration
                    if not (end_time <= assigned_start or start_time >= assigned_end):
                        return False
        return True

    def is_tugboat_available(self, tugboat_id: int, start_time: int, duration: int, 
                           prep_time: int = 0, solution: Solution = None) -> bool:
        """Check if tugboat is available for the given service period including prep time."""
        if solution is None:
            solution = self.current_solution
        
        service_end = start_time + duration + prep_time
        
        for vessel_id in range(self.instance_data['vessel_num']):
            # Check inbound assignments
            for assigned_tug, assigned_start in solution.tugboat_inbound_assignments.get(vessel_id, []):
                if assigned_tug == tugboat_id:
                    assigned_duration = int(self.instance_data['vessel_inbound_service_times'][vessel_id])
                    assigned_prep = int(self.instance_data['inbound_preparation_time'])
                    assigned_end = assigned_start + assigned_duration + assigned_prep
                    
                    if not (service_end <= assigned_start or start_time >= assigned_end):
                        return False
            
            # Check outbound assignments
            for assigned_tug, assigned_start in solution.tugboat_outbound_assignments.get(vessel_id, []):
                if assigned_tug == tugboat_id:
                    assigned_duration = int(self.instance_data['vessel_outbound_service_times'][vessel_id])
                    assigned_prep = int(self.instance_data['outbound_preparation_time'])
                    assigned_end = assigned_start + assigned_duration + assigned_prep
                    
                    if not (service_end <= assigned_start or start_time >= assigned_end):
                        return False
        
        return True

    def find_tugboat_combination(self, vessel_id: int, start_time: int, 
                               service_type: str, solution: Solution = None) -> tuple:
        """Find tugboat combination with sufficient horsepower for the service.
        
        Args:
            vessel_id: ID of the vessel requiring service
            start_time: Start time of the tugboat service
            service_type: 'inbound' or 'outbound'
            solution: Solution to check availability against (default: current_solution)
        
        Returns:
            tuple: (selected_tugboats, total_cost) or (None, 0) if no valid combination found
            selected_tugboats: List of (tugboat_id, start_time) tuples
        """
        if solution is None:
            solution = self.current_solution
            
        required_hp = self.instance_data['vessel_horsepower_requirements'][vessel_id]
        service_duration = (self.instance_data['vessel_inbound_service_times'][vessel_id] if service_type == 'inbound' 
                          else self.instance_data['vessel_outbound_service_times'][vessel_id])
        prep_time = (self.instance_data['inbound_preparation_time'] if service_type == 'inbound' 
                   else self.instance_data['outbound_preparation_time'])
        
        service_duration = int(service_duration)
        prep_time = int(prep_time)
        
        # Get available tugboats for this time slot
        available_tugboats = []
        for tugboat_id in range(self.instance_data['tugboat_num']):
            if self.is_tugboat_available(tugboat_id, start_time, service_duration, prep_time, solution):
                hp = self.instance_data['tugboat_horsepower'][tugboat_id]
                cost = self.instance_data['tugboat_costs'][tugboat_id]
                available_tugboats.append((tugboat_id, hp, cost))
        
        if not available_tugboats:
            return None, 0
        
        total_available_hp = sum(tug[1] for tug in available_tugboats)
        if total_available_hp < required_hp:
            return None, 0
        
        # Greedy: sort by horsepower descending
        available_tugboats.sort(key=lambda x: -x[1])
        
        selected_tugboats = []
        total_hp = 0
        total_cost = 0
        
        for tugboat_id, hp, cost in available_tugboats:
            if len(selected_tugboats) >= self.instance_data['max_tugboats_per_service']:
                break
            
            selected_tugboats.append((tugboat_id, start_time))
            total_hp += hp
            total_cost += cost * service_duration
            
            if total_hp >= required_hp:
                return selected_tugboats, total_cost
        
        # If greedy fails, try exhaustive search
        max_tugs = min(self.instance_data['max_tugboats_per_service'], len(available_tugboats))
        
        for num in range(1, max_tugs + 1):
            for combo in combinations(available_tugboats, num):
                combo_hp = sum(tug[1] for tug in combo)
                if combo_hp >= required_hp:
                    combo_cost = sum(tug[2] * service_duration for tug in combo)
                    combo_tugs = [(tug[0], start_time) for tug in combo]
                    return combo_tugs, combo_cost
        
        return None, 0
            
    def find_feasible_assignments(self, vessel_id: int, max_results: int = 3, 
                                 solution: Solution = None) -> list:
        """
        Find a feasible assignment by trying ETA-centered time points.
        Returns immediately upon finding the first valid assignment.
        """
        if solution is None:
            solution = self.current_solution
    
        vessel_size = self.instance_data['vessel_sizes'][vessel_id]
        eta_i = int(self.instance_data['vessel_etas'][vessel_id])
        duration_i = int(self.instance_data['vessel_durations'][vessel_id])
        tau_in_i = int(self.instance_data['vessel_inbound_service_times'][vessel_id])
        tau_out_i = int(self.instance_data['vessel_outbound_service_times'][vessel_id])
        delta_early_i = int(self.instance_data['vessel_early_limits'][vessel_id])
        delta_late_i = int(self.instance_data['vessel_late_limits'][vessel_id])
        T = self.instance_data['time_periods']
    
        earliest_inbound = max(0, eta_i - delta_early_i)
        latest_inbound = min(T - tau_in_i, eta_i + delta_late_i)
        
        if earliest_inbound > latest_inbound:
            return []
    
        # Get compatible berths
        compatible_berths = []
        for berth_id in range(self.instance_data['berth_num']):
            if self.instance_data['berth_capacities'][berth_id] >= vessel_size:
                compatible_berths.append(berth_id)
        if not compatible_berths:
            return []
    
        # Simple candidate times: ETA and nearby points
        candidate_times = [eta_i]
        for offset in range(1, min(5, max(delta_early_i, delta_late_i) + 1)):
            if eta_i - offset >= earliest_inbound:
                candidate_times.append(eta_i - offset)
            if eta_i + offset <= latest_inbound:
                candidate_times.append(eta_i + offset)
    
        # Try each candidate time
        for t_in in candidate_times:
            inbound_end = t_in + tau_in_i
            if inbound_end > T:
                continue
            
            # Berth starts right after inbound
            t_berth = inbound_end
            berth_end = t_berth + duration_i
            if berth_end > T:
                continue
            
            # Outbound starts right after berth
            t_out = berth_end
            outbound_end = t_out + tau_out_i
            if outbound_end > T:
                continue
            
            # Find tugboat combinations
            inbound_tugboats, _ = self.find_tugboat_combination(
                vessel_id, t_in, 'inbound', solution)
            if inbound_tugboats is None:
                continue
            
            outbound_tugboats, _ = self.find_tugboat_combination(
                vessel_id, t_out, 'outbound', solution)
            
            if outbound_tugboats is None:
                continue
            
            # Check if same tugboats are used for both services
            inbound_tug_ids = {tug_id for tug_id, _ in inbound_tugboats}
            outbound_tug_ids = {tug_id for tug_id, _ in outbound_tugboats}
            
            if inbound_tug_ids & outbound_tug_ids:
                # Check time conflict
                inbound_prep = int(self.instance_data['inbound_preparation_time'])
                if t_in + tau_in_i + inbound_prep > t_out:
                    continue
            
            # Try each compatible berth
            for berth_id in compatible_berths:
                if not self.is_berth_available(berth_id, t_berth, duration_i, solution):
                    continue
                
                # Found a feasible assignment, calculate cost and return
                total_cost = self.calculate_assignment_cost(
                    vessel_id, berth_id, t_berth, 
                    inbound_tugboats, outbound_tugboats)
              
                return [{
                    'berth_id': berth_id,
                    'start_time': t_berth,
                    'inbound_tugboats': inbound_tugboats,
                    'outbound_tugboats': outbound_tugboats,
                    'total_cost': total_cost
                }]
        
        # No feasible assignment found
        return []
        
    def calculate_assignment_cost(self, vessel_id: int, berth_id: int, start_time: int,
                                inbound_tugboats: list, outbound_tugboats: list) -> float:
        """
        Calculate the cost components for a specific vessel assignment with multi-tugboat support.
        
        Args:
            vessel_id: ID of the vessel
            berth_id: ID of the assigned berth
            start_time: Start time of berthing
            inbound_tugboats: List of (tugboat_id, start_time) tuples for inbound service
            outbound_tugboats: List of (tugboat_id, start_time) tuples for outbound service
        
        Returns:
            float: Total assignment cost
        """
        if not inbound_tugboats or not outbound_tugboats:
            return float('inf')  # Invalid assignment
        
        # All tugboats in a service start at the same time (variant 3 constraint)
        inbound_start = inbound_tugboats[0][1]
        outbound_start = outbound_tugboats[0][1]
        
        # Calculate port time cost (Z2 component)
        outbound_duration = self.instance_data['vessel_outbound_service_times'][vessel_id]
        total_port_time = (outbound_start + outbound_duration) - inbound_start
        port_time_cost = (self.instance_data['vessel_priority_weights'][vessel_id] * 
                         self.instance_data['vessel_waiting_costs'][vessel_id] * total_port_time)
        
        # Calculate ETA deviation cost (Z3 component)
        eta = self.instance_data['vessel_etas'][vessel_id]
        if inbound_start >= eta:
            u_early, u_late = 0, inbound_start - eta
        else:
            u_early, u_late = eta - inbound_start, 0
        
        eta_cost = (self.instance_data['vessel_priority_weights'][vessel_id] * 
                   self.instance_data['vessel_jit_costs'][vessel_id] * (u_early + u_late))
        
        # Calculate tugboat utilization cost (Z4 component)
        tugboat_cost = 0.0
        
        # Inbound tugboat costs
        for tugboat_id, _ in inbound_tugboats:
            service_duration = self.instance_data['vessel_inbound_service_times'][vessel_id]
            tugboat_cost += self.instance_data['tugboat_costs'][tugboat_id] * service_duration
        
        # Outbound tugboat costs
        for tugboat_id, _ in outbound_tugboats:
            service_duration = self.instance_data['vessel_outbound_service_times'][vessel_id]
            tugboat_cost += self.instance_data['tugboat_costs'][tugboat_id] * service_duration
        
        return port_time_cost + eta_cost + tugboat_cost

    def helper_function(self) -> dict:
        """Return helper functions."""
        return {
            # Core validation and state
            "get_problem_state": self.get_problem_state,
            "validation_solution": self.validation_solution,
            
            # Basic queries
            "get_unassigned_vessels": self.get_unassigned_vessels,
            "get_vessel_time_window": self.get_vessel_time_window,
            "get_compatible_berths": self.get_compatible_berths,
            "find_tugboat_combination": self.find_tugboat_combination,
            
            # Availability checks
            "is_berth_available": self.is_berth_available,
            "is_tugboat_available": self.is_tugboat_available,
            
            # Assignment functions
            "find_feasible_assignments": self.find_feasible_assignments,  
            "calculate_assignment_cost": self.calculate_assignment_cost,
        }