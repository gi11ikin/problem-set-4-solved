'''
PART 3: SIMILAR ACTROS BY GENRE
Using the imbd_movies dataset:
- Create a data frame, where each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
- Using this data frame as your “feature matrix”, select an actor (called your “query”) for whom you want to find the top 10 most similar actors based on the genres in which they’ve starred 
- - As an example, select the row from your data frame associated with Chris Hemsworth, actor ID “nm1165110”, as your “query” actor
- Use sklearn.metrics.DistanceMetric to calculate the euclidean distances between your query actor and all other actors based on their genre appearances
- - https://scikit-learn.org/stable/modules/generated/sklearn.metrics.DistanceMetric.html
- Output a CSV continaing the top ten actors most similar to your query actor using cosine distance 
- - Name it 'similar_actors_genre_{current_datetime}.csv' to `/data`
- - For example, the top 10 for Chris Hemsworth are:  
        nm1165110 Chris Hemsworth
        nm0000129 Tom Cruise
        nm0147147 Henry Cavill
        nm0829032 Ray Stevenson
        nm5899377 Tiger Shroff
        nm1679372 Sudeep
        nm0003244 Jordi Mollà
        nm0636280 Richard Norton
        nm0607884 Mark Mortimer
        nm2018237 Taylor Kitsch
- Describe in a print() statement how this list changes based on Euclidean distance
- Make sure your code is in line with the standards we're using in this class
'''

#Write your code below
import os
import ast
import pandas as pd
from sklearn.metrics import DistanceMetric
from sklearn.metrics import pairwise
from datetime import datetime

# Ensure /data directory exists
data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(data_dir, exist_ok=True)

csv_path = os.path.join(data_dir, "imbd_movies.csv")

# Load the dataset
df = pd.read_csv(csv_path)

records = []

# df here each row corresponds to an actor, each column represents a genre, and each cell captures how many times that row's actor has appeared in that column’s genre 
for _, row in df.iterrows():
    try:
        actors = ast.literal_eval(row["actors"])
        genres = ast.literal_eval(row["genres"])
    except Exception:
        continue

    for actor_id, actor_name in actors:
        for genre in genres:
            records.append({
                "actor_id": actor_id,
                "actor_name": actor_name,
                "genre": genre
            })

# Actor–genre feature matrix
df_ag = pd.DataFrame(records)
matrix = df_ag.pivot_table(
    index=["actor_id", "actor_name"],
    columns="genre",
    aggfunc=len,
    fill_value=0
)

X = matrix.values
ids = matrix.index

# Select query actor (Chris Hemsworth)
query_id = "nm1165110"
if query_id not in ids.get_level_values(0):
    print("Query actor not found in dataset.")
else:
    q_idx = ids.get_loc(query_id)
    q_vec = X[q_idx].reshape(1, -1)

    # Cosine distance
    distances_cos = pairwise.cosine_distances(q_vec, X)[0]

    pairs_cos = sorted(zip(ids, distances_cos), key=lambda x: x[1])
    top10_cos = [(aid, name, d) for (aid, name), d in pairs_cos if aid != query_id][:10]

    # Save cosine results
    out_path_cos = os.path.join(
        data_dir, f"similar_actors_genre_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    pd.DataFrame(top10_cos, columns=["actor_id", "actor_name", "cosine_distance"]).to_csv(out_path_cos, index=False)

    print("Top 10 actors most similar to Chris Hemsworth (cosine distance):")
    for aid, name, d in top10_cos:
        print(f"{aid} {name} → {d:.4f}")
    print(f"Saved cosine similarity results → {out_path_cos}")

    # Euclidean distance
    dist_euc = DistanceMetric.get_metric("euclidean")
    distances_euc = dist_euc.pairwise(q_vec, X)[0]

    pairs_euc = sorted(zip(ids, distances_euc), key=lambda x: x[1])
    top10_euc = [(aid, name, d) for (aid, name), d in pairs_euc if aid != query_id][:10]

    print("\nTop 10 actors most similar to Chris Hemsworth (euclidean distance):")
    for aid, name, d in top10_euc:
        print(f"{aid} {name} -> {d:.4f}")

    print("\nThe ordering changes between Cosine and Euclidean distance: "
          "Cosine emphasizes similarity in the pattern of genres; "
          "Euclidean emphasizes absolute counts of appearances.")