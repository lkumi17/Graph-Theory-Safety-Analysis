import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from networkx.algorithms.community import girvan_newman

# Load the dataset
df = pd.read_csv('Network analysis dataset.csv')

# Create a graph
G = nx.Graph()

# Add nodes and edges to the graph
for _, row in df.iterrows():
    day = row['Day']
    age = row['Age']
    original_cause_material = row['Original Cause Material']
    injury_type = row['Injury Type']
    accident_time = row['Accident Time']
    accident_month = row['Accident Month']
    company_size = row['Company Size']
    project_scale = row['Project Scale']
    years_of_service = row['Years of Service']
    progress_rate = row['Progress Rate']
    worker_status = row['Worker Status']
    gender = row['Gender']
    pet_range = row['PET range']
    pm10_group = row['PM10_group']

    # Add nodes with attributes
    G.add_node(day, type='Day')
    G.add_node(age, type='Age')
    G.add_node(original_cause_material, type='Original Cause Material')
    G.add_node(injury_type, type='Injury Type')
    G.add_node(accident_time, type='Accident Time')
    G.add_node(accident_month, type='Accident Month')
    G.add_node(company_size, type='Company Size')
    G.add_node(project_scale, type='Project Scale')
    G.add_node(years_of_service, type='Years of Service')
    G.add_node(progress_rate, type='Progress Rate')
    G.add_node(worker_status, type='Worker Status')
    G.add_node(gender, type='Gender')
    G.add_node(pet_range, type='PET range')
    G.add_node(pm10_group, type='PM10 group')

    # Add edges between nodes
    G.add_edge(day, accident_time)
    G.add_edge(day, accident_month)
    G.add_edge(age, years_of_service)
    G.add_edge(age, injury_type)
    G.add_edge(age, project_scale)  
    G.add_edge(age, progress_rate)
    G.add_edge(original_cause_material, injury_type)
    G.add_edge(original_cause_material, accident_time)  
    G.add_edge(original_cause_material, project_scale) 
    G.add_edge(accident_time, project_scale)
    G.add_edge(accident_time, worker_status)  
    G.add_edge(accident_time, gender) 
    G.add_edge(accident_month, project_scale)
    G.add_edge(accident_month, company_size)
    G.add_edge(company_size, progress_rate)
    G.add_edge(company_size, worker_status)
    G.add_edge(company_size, gender)  
    G.add_edge(project_scale, worker_status)
    G.add_edge(years_of_service, progress_rate)
    G.add_edge(years_of_service, worker_status)
    G.add_edge(gender, worker_status)
    G.add_edge(gender, pet_range)
    G.add_edge(pet_range, pm10_group)
    G.add_edge(pet_range, original_cause_material)
    G.add_edge(pet_range, injury_type)
    G.add_edge(pm10_group, company_size)
    G.add_edge(pm10_group, project_scale)

# Streamlit interface
st.title("Construction Safety Factors and Accident Risks Analysis")

# Sidebar for node selection
node_selection = st.sidebar.selectbox("Select a node to simulate removal:", options=list(G.nodes()))

# Define color map for different types
color_map = {
    'Day': '#66c2a5',
    'Age': '#fc8d62',
    'Original Cause Material': '#8da0cb',
    'Injury Type': '#e78ac3',
    'Accident Time': '#a6d854',
    'Accident Month': '#ffd92f',
    'Company Size': '#e5c494',
    'Project Scale': '#b3b3b3',
    'Years of Service': '#1f78b4',
    'Progress Rate': '#33a02c',
    'Worker Status': '#6a3d9a',
    'Gender': '#b15928',
    'PET range': '#ff7f00',
    'PM10 group': '#cab2d6'
}

# Centrality measures
st.sidebar.header("Centrality Measures")
degree_centrality = nx.degree_centrality(G)
betweenness_centrality = nx.betweenness_centrality(G)
closeness_centrality = nx.closeness_centrality(G)

# Display top 20 nodes by centrality measures
def display_top_20_centrality(centrality_dict, centrality_name):
    sorted_centrality = sorted(centrality_dict.items(), key=lambda x: x[1], reverse=True)
    st.sidebar.write(f"Top 20 nodes by {centrality_name}:")
    for node, centrality_value in sorted_centrality[:20]:
        st.sidebar.write(f"Node: {node}, {centrality_name}: {centrality_value:.4f}")

display_top_20_centrality(degree_centrality, "Degree Centrality")
display_top_20_centrality(betweenness_centrality, "Betweenness Centrality")
display_top_20_centrality(closeness_centrality, "Closeness Centrality")

# Simulate node removal and visualize the impact
def simulate_node_removal(graph, node):
    if node in graph:
        graph_copy = graph.copy()
        graph_copy.remove_node(node)

        degree_centrality = nx.degree_centrality(graph_copy)
        betweenness_centrality = nx.betweenness_centrality(graph_copy)
        closeness_centrality = nx.closeness_centrality(graph_copy)

        st.subheader(f"Graph After Removing Node: {node}")
        top_20_degree = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        st.write("Top 20 nodes by Degree Centrality:")
        for node, centrality in top_20_degree:
            st.write(f"Node: {node}, Degree Centrality: {centrality:.4f}")

        top_20_betweenness = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        st.write("Top 20 nodes by Betweenness Centrality:")
        for node, centrality in top_20_betweenness:
            st.write(f"Node: {node}, Betweenness Centrality: {centrality:.4f}")

        top_20_closeness = sorted(closeness_centrality.items(), key=lambda x: x[1], reverse=True)[:20]
        st.write("Top 20 nodes by Closeness Centrality:")
        for node, centrality in top_20_closeness:
            st.write(f"Node: {node}, Closeness Centrality: {centrality:.4f}")

        node_sizes = [1000 * degree_centrality[node] for node in graph_copy.nodes]
        node_colors = [graph_copy.nodes[node]['type'] for node in graph_copy.nodes]
        node_color_values = [color_map.get(node_colors[i], 'lightgrey') for i in range(len(node_colors))]

        plt.figure(figsize=(14, 10))
        pos = nx.kamada_kawai_layout(graph_copy)
        nx.draw(graph_copy, pos, with_labels=True, node_color=node_color_values, node_size=node_sizes, font_size=8, font_color='black', edge_color='gray', alpha=0.7)

        legend_elements = [Line2D([0], [0], marker='o', color='w', label=key, markersize=10, markerfacecolor=value) for key, value in color_map.items()]
        plt.legend(handles=legend_elements, loc='best', fontsize='small')

        st.pyplot(plt)
    else:
        st.write(f"Node {node} not found in the graph.")

# Run the simulation when the user selects a node
if node_selection:
    simulate_node_removal(G, node_selection)
