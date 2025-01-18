import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import matplotlib.pyplot as plt
import umap
import time
import os


def create_folder(root_path, directory_separator="/", next_path="next"):
    output_dir = root_path + directory_separator + next_path
    try:
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

    except Exception as e:
        print(e)

        exit(1)
    return output_dir


def umap_func(embeddings, n_components=3):

    start = time.time()
    projections = umap.UMAP(n_components=n_components).fit_transform(embeddings)
    end = time.time()
    print(f"generating projections with UMAP took: {(end-start):.2f} sec")
    return projections


def plot_feature_counter(counter_df):

    # Plot the bar chart
    plt.figure(figsize=(12, 8))
    plt.bar(counter_df["Feature"], counter_df["Count"], color="skyblue")

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=90, fontsize=10)

    # Add title and labels
    plt.title("Feature Counts", fontsize=14)
    plt.xlabel("Feature", fontsize=12)
    plt.ylabel("Count", fontsize=12)

    # Show the plot
    plt.tight_layout()  # Adjusts the layout to make space for the labels
    plt.show()


def extract_projections(projections_2d):
    first_projection = list(projections_2d[:, 0])
    second_projection = list(projections_2d[:, 1])
    return first_projection, second_projection


def umap_2d_scatter_chart(df, model_embedding_name):
    # Create the scatter plot
    fig = px.scatter(
        df,
        x="umap_2d_first",
        y="umap_2d_second",
        color="goal",  # color points based on 'goal' categories
        title="UMAP 2D Projection with Goal Categories and Descriptions",
        hover_data={
            "name": True,
            "goal": True,
            "umap_2d_first": False,
            "umap_2d_second": False,
        },
        # text="name",
        labels={
            "umap_2d_first": "UMAP 2D X",
            "umap_2d_second": "UMAP 2D Y",
        },  # for axis
        width=1200,
        height=800,
    )

    # Show the plot
    # fig.show()
    fig.write_html("out/umap2d_with_class_" + model_embedding_name + ".html")
