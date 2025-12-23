import networkx as nx
import numpy as np
import random
from flask import Flask, jsonify, request
from flask_cors import CORS
import json

app = Flask(__name__)
CORS(app)

class DiseaseSimulation:
    def __init__(self, num_nodes=4000):
        self.num_nodes = num_nodes
        self.G = nx.Graph()
        self.states = {}  # node_id -> state (S, E, I, R)
        self.state_timers = {}  # node_id -> time remaining in current state
        self.history = []
        self.current_step = 0
        
        # Simulation parameters
        self.EXPOSED_DURATION = 3  # steps before becoming infected
        self.INFECTED_DURATION = 6  # steps before recovering
        
        # Transmission probabilities by context
        self.TRANS_PROB = {
            'household': 0.25,
            'workplace': 0.12,
            'community': 0.06,
            'random': 0.03
        }
        
    def generate_network(self):
        """Generate multi-layered social network"""
        node_id = 0
        
        # 1. Household clusters (750 households, 4-6 people each)
        num_households = 750
        for _ in range(num_households):
            household_size = random.randint(4, 6)
            household_nodes = list(range(node_id, node_id + household_size))
            
            # Complete graph for household
            for i in household_nodes:
                for j in household_nodes:
                    if i < j:
                        self.G.add_edge(i, j, context='household', weight=1.0)
            
            node_id += household_size
        
        # 2. Workplace/School networks (Watts-Strogatz small-world)
        workplace_size = 50
        num_workplaces = self.num_nodes // workplace_size
        
        for w in range(num_workplaces):
            start = w * workplace_size
            end = min(start + workplace_size, self.num_nodes)
            workplace_nodes = list(range(start, end))
            
            if len(workplace_nodes) >= 10:
                # Create small-world network
                k = 6  # each node connected to k nearest neighbors
                p = 0.1  # rewiring probability
                
                for i, node in enumerate(workplace_nodes):
                    for j in range(1, k // 2 + 1):
                        neighbor_idx = (i + j) % len(workplace_nodes)
                        neighbor = workplace_nodes[neighbor_idx]
                        
                        if random.random() > p:
                            self.G.add_edge(node, neighbor, context='workplace', weight=0.7)
                        else:
                            # Rewire to random node
                            random_node = random.choice(workplace_nodes)
                            if random_node != node:
                                self.G.add_edge(node, random_node, context='workplace', weight=0.7)
        
        # 3. Community layer (Barabási-Albert scale-free)
        # Add preferential attachment edges
        m = 3  # number of edges to attach
        targets = list(range(m))
        
        for node in range(m, self.num_nodes):
            # Preferential attachment
            degrees = dict(self.G.degree())
            total_degree = sum(degrees.values()) + len(degrees)
            
            probs = [(degrees.get(t, 0) + 1) / total_degree for t in range(node)]
            probs = np.array(probs)
            probs = probs / probs.sum()
            
            new_targets = np.random.choice(node, size=min(m, node), replace=False, p=probs)
            
            for target in new_targets:
                self.G.add_edge(node, int(target), context='community', weight=0.5)
        
        # 4. Random encounters (sparse Erdős-Rényi)
        num_random_edges = self.num_nodes * 2
        for _ in range(num_random_edges):
            u = random.randint(0, self.num_nodes - 1)
            v = random.randint(0, self.num_nodes - 1)
            if u != v and not self.G.has_edge(u, v):
                self.G.add_edge(u, v, context='random', weight=0.3)
        
        print(f"Network generated: {self.G.number_of_nodes()} nodes, {self.G.number_of_edges()} edges")
        
    def initialize_simulation(self, initial_infected=7):
        """Initialize all nodes as susceptible, then infect random nodes"""
        for node in self.G.nodes():
            self.states[node] = 'S'
            self.state_timers[node] = 0
        
        # Randomly infect initial nodes
        infected_nodes = random.sample(list(self.G.nodes()), initial_infected)
        for node in infected_nodes:
            self.states[node] = 'I'
            self.state_timers[node] = self.INFECTED_DURATION
        
        self.current_step = 0
        self.history = [self._get_current_state_snapshot()]
        
    def _get_current_state_snapshot(self):
        """Get current state counts"""
        counts = {'S': 0, 'E': 0, 'I': 0, 'R': 0}
        for state in self.states.values():
            counts[state] += 1
        
        return {
            'step': self.current_step,
            'counts': counts,
            'states': dict(self.states)
        }
    
    def _get_active_contexts(self, step):
        """Determine which network contexts are active at this time step"""
        hour = (step * 4) % 24  # Each step = 4 hours
        
        active = ['household', 'random']  # Always active
        
        if 8 <= hour < 18:  # 8am - 6pm
            active.append('workplace')
        else:  # Evenings
            active.append('community')
        
        return active
    
    def step_simulation(self):
        """Advance simulation by one time step"""
        active_contexts = self._get_active_contexts(self.current_step)
        new_exposed = []
        
        # 1. Disease transmission
        infected_nodes = [n for n, s in self.states.items() if s == 'I']
        
        for infected_node in infected_nodes:
            for neighbor in self.G.neighbors(infected_node):
                if self.states[neighbor] == 'S':
                    # Check edge context
                    edge_data = self.G.get_edge_data(infected_node, neighbor)
                    context = edge_data.get('context', 'random')
                    
                    if context in active_contexts:
                        trans_prob = self.TRANS_PROB[context]
                        
                        if random.random() < trans_prob:
                            new_exposed.append(neighbor)
        
        # Apply new exposures
        for node in new_exposed:
            if self.states[node] == 'S':  # Double check still susceptible
                self.states[node] = 'E'
                self.state_timers[node] = self.EXPOSED_DURATION
        
        # 2. Update state timers and transitions
        for node in list(self.states.keys()):
            if self.states[node] == 'E':
                self.state_timers[node] -= 1
                if self.state_timers[node] <= 0:
                    self.states[node] = 'I'
                    self.state_timers[node] = self.INFECTED_DURATION
            
            elif self.states[node] == 'I':
                self.state_timers[node] -= 1
                if self.state_timers[node] <= 0:
                    self.states[node] = 'R'
                    self.state_timers[node] = 0
        
        self.current_step += 1
        self.history.append(self._get_current_state_snapshot())
        
        # Check if simulation should stop
        infected_count = sum(1 for s in self.states.values() if s == 'I')
        exposed_count = sum(1 for s in self.states.values() if s == 'E')
        
        return infected_count > 0 or exposed_count > 0
    
    def run_full_simulation(self, max_steps=100):
        """Run simulation until completion or max steps"""
        while self.current_step < max_steps:
            should_continue = self.step_simulation()
            if not should_continue:
                break
        
        return self.history
    
    def get_network_data(self):
        """Export network data for visualization"""
        nodes = []
        for node in self.G.nodes():
            nodes.append({
                'id': node,
                'state': self.states.get(node, 'S')
            })
        
        edges = []
        for u, v, data in self.G.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'context': data.get('context', 'random')
            })
        
        return {'nodes': nodes, 'edges': edges}
    
    def calculate_r0(self):
        """Calculate basic reproduction number"""
        if len(self.history) < 10:
            return 0
        
        # Simple R0 estimation: average new infections per infected individual
        early_steps = self.history[:10]
        total_new_infections = 0
        total_infected_person_steps = 0
        
        for i in range(1, len(early_steps)):
            prev_counts = early_steps[i-1]['counts']
            curr_counts = early_steps[i]['counts']
            
            new_infections = curr_counts['E'] - prev_counts['E'] + curr_counts['I'] - prev_counts['I']
            if new_infections > 0:
                total_new_infections += new_infections
                total_infected_person_steps += prev_counts['I']
        
        if total_infected_person_steps > 0:
            return total_new_infections / total_infected_person_steps
        return 0

# Global simulation instance
sim = None

@app.route('/api/initialize', methods=['POST'])
def initialize():
    global sim
    data = request.json
    num_nodes = data.get('num_nodes', 4000)
    initial_infected = data.get('initial_infected', 7)
    
    sim = DiseaseSimulation(num_nodes)
    sim.generate_network()
    sim.initialize_simulation(initial_infected)
    
    return jsonify({
        'success': True,
        'message': f'Simulation initialized with {num_nodes} nodes',
        'initial_state': sim.history[0]
    })

@app.route('/api/step', methods=['POST'])
def step():
    global sim
    if sim is None:
        return jsonify({'error': 'Simulation not initialized'}), 400
    
    should_continue = sim.step_simulation()
    
    return jsonify({
        'success': True,
        'should_continue': should_continue,
        'current_state': sim.history[-1],
        'step': sim.current_step
    })

@app.route('/api/run', methods=['POST'])
def run_full():
    global sim
    if sim is None:
        return jsonify({'error': 'Simulation not initialized'}), 400
    
    data = request.json
    max_steps = data.get('max_steps', 100)
    
    history = sim.run_full_simulation(max_steps)
    r0 = sim.calculate_r0()
    
    return jsonify({
        'success': True,
        'history': history,
        'r0': r0,
        'total_steps': sim.current_step
    })

@app.route('/api/network', methods=['GET'])
def get_network():
    global sim
    if sim is None:
        return jsonify({'error': 'Simulation not initialized'}), 400
    
    return jsonify(sim.get_network_data())

@app.route('/api/status', methods=['GET'])
def get_status():
    global sim
    if sim is None:
        return jsonify({'initialized': False})
    
    return jsonify({
        'initialized': True,
        'num_nodes': sim.num_nodes,
        'num_edges': sim.G.number_of_edges(),
        'current_step': sim.current_step,
        'current_state': sim.history[-1] if sim.history else None
    })

if __name__ == '__main__':
    print("Starting Disease Simulation Backend...")
    print("Server running on http://localhost:5000")
    app.run(debug=True, port=5000)