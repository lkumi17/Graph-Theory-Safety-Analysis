import streamlit as st
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from networkx.algorithms.community import girvan_newman

# Load the dataset
df = pd.read_csv('translated_file.csv')

# Extract relevant columns for analysis
columns_of_interest = [
    '공사규모', '발생형태', '기인물(대)', '기인물(중)', '기인물(소)',
    '소업종명', '규모', '직종', '연령', '성별', '근속기간', '재해요일',
    '재해시간', '재해개요', '시설물(대)', '근로자수'
]
df_filtered = df[columns_of_interest]
df_filtered.dropna(subset=['공사규모', '발생형태', '기인물(대)', '소업종명'], inplace=True)

# Create a graph
G = nx.Graph()

# Add nodes and edges to the graph
for _, row in df_filtered.iterrows():
    project_scale = row['공사규모']
    accident_type = row['발생형태']
    major_cause = row['기인물(대)']
    medium_cause = row['기인물(중)']
    minor_cause = row['기인물(소)']
    industry_type = row['소업종명']
    occupation_type = row['직종']
    age = row['연령']
    gender = row['성별']
    experience = row['근속기간']
    weekday = row['재해요일']
    time = row['재해시간']
    facility = row['시설물(대)']
    worker_count = row['근로자수']

    G.add_node(project_scale, type='ProjectScale')
    G.add_node(accident_type, type='AccidentType')
    G.add_node(major_cause, type='MajorCause')
    G.add_node(medium_cause, type='MediumCause')
    G.add_node(minor_cause, type='MinorCause')
    G.add_node(industry_type, type='IndustryType')
    G.add_node(occupation_type, type='OccupationType')
    G.add_node(age, type='Age')
    G.add_node(gender, type='Gender')
    G.add_node(experience, type='Experience')
    G.add_node(weekday, type='Weekday')
    G.add_node(time, type='Time')
    G.add_node(facility, type='Facility')
    G.add_node(worker_count, type='WorkerCount')

    G.add_edge(project_scale, accident_type)
    G.add_edge(accident_type, major_cause)
    G.add_edge(major_cause, medium_cause)
    G.add_edge(medium_cause, minor_cause)
    G.add_edge(minor_cause, industry_type)
    G.add_edge(industry_type, occupation_type)
    G.add_edge(occupation_type, age)
    G.add_edge(age, gender)
    G.add_edge(gender, experience)
    G.add_edge(experience, weekday)
    G.add_edge(weekday, time)
    G.add_edge(time, facility)
    G.add_edge(facility, worker_count)

# Streamlit interface
st.title("Construction Safety Factors and Accident Risks Analysis")

# Sidebar for node selection
node_selection = st.sidebar.selectbox("Select a node to simulate removal:", options=list(G.nodes()))

# Define color map for different types
color_map = {
    'ProjectScale': '#66c2a5',
    'AccidentType': '#fc8d62',
    'MajorCause': '#8da0cb',
    'MediumCause': '#e78ac3',
    'MinorCause': '#a6d854',
    'IndustryType': '#ffd92f',
    'OccupationType': '#e5c494',
    'Age': '#b3b3b3',
    'Gender': '#1f78b4',
    'Experience': '#33a02c',
    'Weekday': '#6a3d9a',
    'Time': '#b15928',
    'Facility': '#ff7f00',
    'WorkerCount': '#cab2d6'
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
