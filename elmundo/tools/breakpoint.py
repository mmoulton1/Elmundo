from capital_costs import capital_cost_salt_cavern, capital_cost_underground_pipes, Compressor, pipeline, capital_cost_hardrock_cavern
import scipy.optimize as opt

def combined_cost_salt_cavern_compressor_pipeline(storage_capacity, p_outlet, flow_rate_kg_d, pipeline_length):
    """
    Calculate the combined capital cost of H2 storage in salt caverns, compressors, and pipelines.

    Parameters:
    storage_capacity (float): Storage capacity of the cavern in kg.
    p_outlet (float): Outlet pressure for the compressor in bar.
    flow_rate_kg_d (float): Flow rate for the compressor in kg/day.
    pipeline_length (float): Length of the pipeline in km.

    Returns:
    dict: Dictionary containing individual and total capital costs in 2018 USD (actually I'm not sure what year dollars it is in).
    """

    # Calculate capital cost for salt cavern storage
    cavern_cost, cavern_cost_per_kg_H2 = capital_cost_salt_cavern(storage_capacity)

    # Initialize compressor
    compressor = Compressor(p_outlet=p_outlet, flow_rate_kg_d=flow_rate_kg_d)

    # Calculate compressor power
    compressor.compressor_power()
    compressor_power, system_power = compressor.compressor_system_power()

    # Calculate compressor costs
    compressor_cost = compressor.compressor_costs()

    # Calculate capital cost for pipeline
    pipeline_cost = pipeline(pipeline_length)

    # Combine costs
    total_cost = cavern_cost + compressor_cost + pipeline_cost
    overall_cost_per_kg_H2 = total_cost/storage_capacity

    return {
        'cavern_cost': cavern_cost,
        'cavern_cost_per_kg_H2': cavern_cost_per_kg_H2,
        'compressor_power': compressor_power,
        'system_power': system_power,
        'compressor_cost': compressor_cost,
        'pipeline_cost': pipeline_cost,
        'total_cost': total_cost,
        'overall_cost_per_kg_H2' : overall_cost_per_kg_H2
        
        
    }

# Example usage
if __name__ == "__main__":
    storage_capacity = 6000000  # Example storage capacity in kg
    p_outlet = 100  # Example outlet pressure in bar
    flow_rate_kg_d = 50000  # Example flow rate in kg/day
    pipeline_length = 1000  # Example pipeline length in km

    costs = combined_cost_salt_cavern_compressor_pipeline(storage_capacity, p_outlet, flow_rate_kg_d, pipeline_length)
    print(costs)

def combined_cost_hardrock_cavern_compressor_pipeline(storage_capacity, p_outlet, flow_rate_kg_d, pipeline_length):
    """
    Calculate the combined capital cost of H2 storage in hardrock caverns, compressors, and pipelines.

    Parameters:
    storage_capacity (float): Storage capacity of the cavern in kg.
    p_outlet (float): Outlet pressure for the compressor in bar.
    flow_rate_kg_d (float): Flow rate for the compressor in kg/day.
    pipeline_length (float): Length of the pipeline in km.

    Returns:
    dict: Dictionary containing individual and total capital costs in 2018 USD (actually I'm not sure what year dollars it is in).
    """

    # Calculate capital cost for salt cavern storage
    cavern_cost, cavern_cost_per_kg_H2 = capital_cost_hardrock_cavern(storage_capacity)

    # Initialize compressor
    compressor = Compressor(p_outlet=p_outlet, flow_rate_kg_d=flow_rate_kg_d)

    # Calculate compressor power
    compressor.compressor_power()
    compressor_power, system_power = compressor.compressor_system_power()

    # Calculate compressor costs
    compressor_cost = compressor.compressor_costs()

    # Calculate capital cost for pipeline
    pipeline_cost = pipeline(pipeline_length)

    # Combine costs
    total_cost = cavern_cost + compressor_cost + pipeline_cost
    overall_cost_per_kg_H2 = total_cost/storage_capacity

    return {
        'cavern_cost': cavern_cost,
        'cavern_cost_per_kg_H2': cavern_cost_per_kg_H2,
        'compressor_power': compressor_power,
        'system_power': system_power,
        'compressor_cost': compressor_cost,
        'pipeline_cost': pipeline_cost,
        'total_cost': total_cost,
        'overall_cost_per_kg_H2' : overall_cost_per_kg_H2
        
        
    }

# Example usage
if __name__ == "__main__":
    storage_capacity = 6000000  # Example storage capacity in kg
    p_outlet = 100  # Example outlet pressure in bar
    flow_rate_kg_d = 50000  # Example flow rate in kg/day
    pipeline_length = 1000  # Example pipeline length in km

    costs = combined_cost_hardrock_cavern_compressor_pipeline(storage_capacity, p_outlet, flow_rate_kg_d, pipeline_length)
    print(costs)
 

def combined_cost_underground_pipe_compressor(storage_capacity, p_outlet, flow_rate_kg_d):
    """
    Calculate the combined capital cost of H2 storage in underground pipes and compressors.

    Parameters:
    storage_capacity (float): Storage capacity of the pipes in kg.
    p_outlet (float): Outlet pressure for the compressor in bar.
    flow_rate_kg_d (float): Flow rate for the compressor in kg/day.

    Returns:
    dict: Dictionary containing individual and total capital costs in 2018 USD.
    """

    # Calculate capital cost for underground pipe storage
    pipe_cost, pipe_cost_per_kg_H2 = capital_cost_underground_pipes(storage_capacity)

    # Initialize compressor
    compressor = Compressor(p_outlet=p_outlet, flow_rate_kg_d=flow_rate_kg_d)

    # Calculate compressor power
    compressor.compressor_power()
    compressor_power, system_power = compressor.compressor_system_power()

    # Calculate compressor costs
    compressor_cost = compressor.compressor_costs()

    # Combine costs
    total_cost = pipe_cost + compressor_cost
    overall_cost_per_kg_H2 = total_cost/storage_capacity

    return {
        'pipe_cost': pipe_cost,
        'pipe_cost_per_kg_H2': pipe_cost_per_kg_H2,
        'compressor_power': compressor_power,
        'system_power': system_power,
        'compressor_cost': compressor_cost,
        'total_cost': total_cost,
        'overall_cost_per_kg_H2' : overall_cost_per_kg_H2
    }

# Example usage
if __name__ == "__main__":
    storage_capacity = 6000000  # Example storage capacity in kg
    p_outlet = 100  # Example outlet pressure in bar
    flow_rate_kg_d = 50000  # Example flow rate in kg/day

    costs = combined_cost_underground_pipe_compressor(storage_capacity, p_outlet, flow_rate_kg_d)
    print(costs)

def find_equilibrium_pipeline_length(storage_capacity, p_outlet, flow_rate_kg_d):
    """
    Find the pipeline length where the combined costs of both methods are equal.

    Parameters:
    storage_capacity (float): Storage capacity of the cavern or pipes in kg.
    p_outlet (float): Outlet pressure for the compressor in bar.
    flow_rate_kg_d (float): Flow rate for the compressor in kg/day.

    Returns:
    float: Pipeline length in km where the combined costs are equal.
    """
    def cost_difference(pipeline_length):
        cost_salt_cavern = combined_cost_salt_cavern_compressor_pipeline(storage_capacity, p_outlet, flow_rate_kg_d, pipeline_length)['total_cost']
        cost_underground_pipe = combined_cost_underground_pipe_compressor(storage_capacity, p_outlet, flow_rate_kg_d)['total_cost']
        return cost_salt_cavern - cost_underground_pipe

    # Initial bracket range
    bracket = [0, 1000]
    f_a = cost_difference(bracket[0])
    f_b = cost_difference(bracket[1])

    # Adjust the bracket if the signs are not different
    while f_a * f_b > 0:
        bracket[1] *= 1.5  # Increase the upper bound
        f_b = cost_difference(bracket[1])

    result = opt.root_scalar(cost_difference, bracket=bracket, method='brentq')
    return result.root

# Example usage
if __name__ == "__main__":
    storage_capacity = 1500000  # Example storage capacity in kg
    p_outlet = 100  # Example outlet pressure in bar
    flow_rate_kg_d = 439000  # Example flow rate in kg/day

    equilibrium_pipeline_length = find_equilibrium_pipeline_length(storage_capacity, p_outlet, flow_rate_kg_d)
    print(f"Equilibrium pipeline length: {equilibrium_pipeline_length:.2f} km")

import numpy as np
import matplotlib.pyplot as plt

# Define parameters
p_outlet = 100  # Example outlet pressure in bar
flow_rate_kg_d = 50000  # Example flow rate in kg/day
pipeline_length = 300
# Define a range of storage capacities
storage_capacities = np.linspace(100000, 10000000, 100)  # Example range from 100,000 to 10,000,000 kg

# Calculate costs for the salt cavern + compressor + pipeline function
costs_salt_cavern_compressor_pipeline = []
for storage_capacity in storage_capacities:
    cost_salt_cavern_compressor_pipeline = combined_cost_salt_cavern_compressor_pipeline(storage_capacity, p_outlet, flow_rate_kg_d, pipeline_length)['total_cost']
    costs_salt_cavern_compressor_pipeline.append(cost_salt_cavern_compressor_pipeline)

# Calculate costs for the underground pipe + compressor function
costs_underground_pipe_compressor = []
for storage_capacity in storage_capacities:
    cost_underground_pipe_compressor = combined_cost_underground_pipe_compressor(storage_capacity, p_outlet, flow_rate_kg_d)['total_cost']
    costs_underground_pipe_compressor.append(cost_underground_pipe_compressor)

# Convert to numpy arrays for plotting
costs_salt_cavern_compressor_pipeline = np.array(costs_salt_cavern_compressor_pipeline)
costs_underground_pipe_compressor = np.array(costs_underground_pipe_compressor)


# Plotting
plt.figure(figsize=(10, 6))

plt.plot(storage_capacities, costs_salt_cavern_compressor_pipeline, label='Salt Cavern + Compressor + Pipeline')
plt.plot(storage_capacities, costs_underground_pipe_compressor, label='Underground Pipe + Compressor')

plt.xlabel('Storage Capacity (kg)')
plt.ylabel('Total Capital Cost (USD)')
plt.title('Capital Costs Comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Constants (keeping other parameters constant)
storage_capacity = 6000000  # Example storage capacity in kg
p_outlet = 100  # Example outlet pressure in bar
flow_rate_kg_d = 50000  # Example flow rate in kg/day

# Vary pipeline lengths for both scenarios
pipeline_lengths = np.linspace(10, 800, 50)  # Example range from 10 to 500 km

# Calculate costs for underground pipe + compressor
costs_underground_pipe_compressor = []
for length in pipeline_lengths:
    cost = combined_cost_underground_pipe_compressor(storage_capacity, p_outlet, flow_rate_kg_d)['total_cost']
    costs_underground_pipe_compressor.append(cost)

# Convert to numpy array for plotting
costs_underground_pipe_compressor = np.array(costs_underground_pipe_compressor)

# Calculate costs for salt cavern + compressor + pipeline
costs_salt_cavern_compressor_pipeline = []
for length in pipeline_lengths:
    cost = combined_cost_salt_cavern_compressor_pipeline(storage_capacity, p_outlet, flow_rate_kg_d, length)['total_cost']
    costs_salt_cavern_compressor_pipeline.append(cost)

# Convert to numpy array for plotting
costs_salt_cavern_compressor_pipeline = np.array(costs_salt_cavern_compressor_pipeline)

# Plotting
plt.figure(figsize=(10, 6))

plt.plot(pipeline_lengths, costs_underground_pipe_compressor, label='Underground Pipe + Compressor')
plt.plot(pipeline_lengths, costs_salt_cavern_compressor_pipeline, label='Salt Cavern + Compressor + Pipeline')

plt.xlabel('Pipeline Length (km)')
plt.ylabel('Total Capital Cost (USD)')
plt.title('Capital Cost vs Pipeline Length')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Define the range of storage capacities
storage_capacities = np.linspace(100000, 8000000, 50000)  # Example range from 1,000 to 500,000 kg

# Fixed parameters
p_outlet = 100
flow_rate_kg_d = 439481
pipeline_length = 0

# Initialize lists to store results
salt_cavern_costs = []
underground_pipe_costs = []

# Compute the costs for each storage capacity
for capacity in storage_capacities:
    salt_cavern_result = combined_cost_salt_cavern_compressor_pipeline(capacity, p_outlet, flow_rate_kg_d, pipeline_length)
    underground_pipe_result = combined_cost_underground_pipe_compressor(capacity, p_outlet, flow_rate_kg_d)
    
    salt_cavern_costs.append(salt_cavern_result['overall_cost_per_kg_H2'])
    underground_pipe_costs.append(underground_pipe_result['overall_cost_per_kg_H2'])

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(storage_capacities, salt_cavern_costs, label='Salt Cavern Storage', color='b')
plt.plot(storage_capacities, underground_pipe_costs, label='Underground Pipe Storage', color='r')
plt.xlabel('Storage Capacity (kg)')
plt.ylabel('Overall Cost per kg H2 (USD)')
plt.title('Overall Cost per kg H2 vs Storage Capacity')
plt.legend()
plt.grid(True)
plt.show()