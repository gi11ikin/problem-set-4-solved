'''
PART 2: NETWORK CENTRALITY METRICS

Using the imbd_movies dataset
- Guild a graph and perform some rudimentary graph analysis, extracting centrality metrics from it. 
- Below is some basic code scaffolding that you will need to add to. 
- Tailor this code scaffolding and its stucture to however works to answer the problem
- Make sure the code is line with the standards we're using in this class 
'''

import numpy as np
import pandas as pd
import networkx as nx
import os
from datetime import datetime
import ast

# Build the graph
g = nx.Graph()

# Set up your dataframe(s) -> the df that's output to a CSV should include at least the columns 'left_actor_name', '<->', 'right_actor_name'
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, "imbd_movies.csv")

edges = []

# Read the CSV from Part 1
df = pd.read_csv(csv_path)

# Iterate through rows, each row has an "actors" column (JSON string)
for _, row in df.iterrows():
    try:
        actors = ast.literal_eval(row["actors"])   # parse the list safely
    except Exception:
        continue

    # Add all actors as nodes
    for actor_id, actor_name in actors:
        g.add_node(actor_id, name=actor_name)

    # Generate all unique pairs of actors in this movie
    for i in range(len(actors)):
        left_actor_id, left_actor_name = actors[i]
        for j in range(i+1, len(actors)):
            right_actor_id, right_actor_name = actors[j]

            if g.has_edge(left_actor_id, right_actor_id):
                g[left_actor_id][right_actor_id]["weight"] += 1
            else:
                g.add_edge(left_actor_id, right_actor_id, weight=1)

            edges.append({
                "left_actor_name": left_actor_name,
                "<->": "<->",
                "right_actor_name": right_actor_name
            })

# Print the info below
print("Nodes:", len(g.nodes))

# Print the 10 most central nodes
centrality = nx.degree_centrality(g)
top10 = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:10]
print("Top 10 most central actors:")
for actor_id, score in top10:
    print(f"{g.nodes[actor_id]['name']} ({actor_id}) → {score:.4f}")

# Output the final dataframe to a CSV named 'network_centrality_{current_datetime}.csv' to `/data`
edges_df = pd.DataFrame(edges)
out_path = os.path.join(
    data_dir, f"network_centrality_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
)
edges_df.to_csv(out_path, index=False)