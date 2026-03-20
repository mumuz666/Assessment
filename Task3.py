import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from openai import OpenAI
client = OpenAI(api_key="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")

def generate_ai_insight(prompt_text):
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a real estate data analyst. Provide concise, professional insights."},
                {"role": "user", "content": prompt_text}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error generating insight: {e}"
    
st.set_page_config(page_title="Real Estate Lease-Up Dashboard", layout="wide")

# Title and description
st.title("GenAI-Enhanced Real Estate Lease-Up Dashboard")
st.markdown("Interactive dashboard based on Task 1 feature engineering results.")

# Load data
df = pd.read_csv(r"C:\Users\联想\Desktop\笔试\Task3\final_features.csv")

# Convert date columns to datetime
if "FirstRecordedMonth" in df.columns:
    df["FirstRecordedMonth"] = pd.to_datetime(df["FirstRecordedMonth"], errors="coerce")
if "FirstMonthOcc90" in df.columns:
    df["FirstMonthOcc90"] = pd.to_datetime(df["FirstMonthOcc90"], errors="coerce")

# Clustering + PCA preparation
cluster_features = [
    "InitialOccupancy",
    "AvgMonthlyOccGrowth",
    "TimeTo50Pct",
    "OccDropIndicator",
    "OccupancyVolatility",
    "RentVolatility"
]

available_cluster_features = [c for c in cluster_features if c in df.columns]

if len(available_cluster_features) >= 2:
    cluster_df = df.dropna(subset=available_cluster_features).copy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(cluster_df[available_cluster_features])

    # KMeans clustering
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    cluster_df["cluster"] = kmeans.fit_predict(X_scaled)

    # PCA (2D)
    pca = PCA(n_components=2, random_state=42)
    pca_coords = pca.fit_transform(X_scaled)

    cluster_df["PC1"] = pca_coords[:, 0]
    cluster_df["PC2"] = pca_coords[:, 1]

    # merge back to original df
    df = df.merge(
        cluster_df[["PropertyID", "cluster", "PC1", "PC2"]],
        on="PropertyID",
        how="left"
    )

# Sidebar filters
st.sidebar.header("Filters")

market_list = ["All"] + sorted(df["MarketName"].dropna().unique().tolist())
selected_market = st.sidebar.selectbox("Select Market", market_list)

if selected_market == "All":
    filtered_df = df.copy()
else:
    filtered_df = df[df["MarketName"] == selected_market].copy()

# Data preview
st.subheader("Data Preview")
st.dataframe(filtered_df)

# KPI
st.subheader("Key Metrics")
col1, col2, col3 = st.columns(3)

col1.metric("Number of Properties", len(filtered_df))

if "LeaseUpMonths" in filtered_df.columns:
    avg_lease = filtered_df["LeaseUpMonths"].mean()
    col2.metric("Average Lease-Up Months", f"{avg_lease:.2f}" if pd.notna(avg_lease) else "N/A")
else:
    col2.metric("Average Lease-Up Months", "N/A")

if "NegativeEffRentGrowth" in filtered_df.columns:
    neg_share = filtered_df["NegativeEffRentGrowth"].mean()
    col3.metric("Negative Rent Growth Share", f"{neg_share:.1%}" if pd.notna(neg_share) else "N/A")
else:
    col3.metric("Negative Rent Growth Share", "N/A")

# Lease-Up Months Distribution
if "LeaseUpMonths" in filtered_df.columns:
    st.subheader("Lease-Up Months Distribution")
    fig1 = px.histogram(
        filtered_df,
        x="LeaseUpMonths",
        nbins=20,
        title="Lease-Up Months Distribution"
    )
    fig1.update_layout(
        xaxis_title="Lease-Up Months",
        yaxis_title="Count"
    )
    st.plotly_chart(fig1, use_container_width=True)

# AI Insight for Distribution
if st.button("Generate Insight for Distribution"):
    prompt = f"""
    Analyze the lease-up distribution:
    Mean: {filtered_df['LeaseUpMonths'].mean():.2f}
    Median: {filtered_df['LeaseUpMonths'].median():.2f}
    Max: {filtered_df['LeaseUpMonths'].max():.2f}
    Min: {filtered_df['LeaseUpMonths'].min():.2f}

    Provide insights about distribution shape and implications.
    """

    st.write(generate_ai_insight(prompt))

# Initial Occupancy vs Avg Monthly Occupancy Growth
if "InitialOccupancy" in filtered_df.columns and "AvgMonthlyOccGrowth" in filtered_df.columns:
    st.subheader("Initial Occupancy vs Avg Monthly Occupancy Growth")
    fig2 = px.scatter(
        filtered_df,
        x="InitialOccupancy",
        y="AvgMonthlyOccGrowth",
        hover_data=["PropertyID", "Name", "MarketName", "LeaseUpMonths"],
        title="Initial Occupancy vs Avg Monthly Occupancy Growth"
    )
    fig2.update_layout(
        xaxis_title="Initial Occupancy",
        yaxis_title="Avg Monthly Occupancy Growth"
    )
    st.plotly_chart(fig2, use_container_width=True)

# AI Insight for Occupancy Growth
if st.button("Generate Insight for Occupancy Growth"):
    prompt = f"""
    Analyze relationship between initial occupancy and growth.

    Avg Initial Occupancy: {filtered_df['InitialOccupancy'].mean():.2f}
    Avg Growth: {filtered_df['AvgMonthlyOccGrowth'].mean():.2f}

    Explain relationship and any pattern.
    """

    st.write(generate_ai_insight(prompt))

# Market comparison
if "MarketName" in df.columns and "LeaseUpMonths" in df.columns:
    st.subheader("Average Lease-Up Months by Market")
    market_summary = (
        df.groupby("MarketName", dropna=False)["LeaseUpMonths"]
        .mean()
        .reset_index()
        .sort_values("LeaseUpMonths")
    )

    fig3 = px.bar(
        market_summary,
        x="LeaseUpMonths",
        y="MarketName",
        orientation="h",
        hover_data=["LeaseUpMonths"],
        title="Average Lease-Up Months by Market"
    )

    fig3.update_layout(
        xaxis_title="Average Lease-Up Months",
        yaxis_title="Market"
    )

    st.plotly_chart(fig3, use_container_width=True)

# AI Insight for Market Comparison
if st.button("Generate Insight for Market"):
    market_stats = market_summary.to_string()

    prompt = f"""
    Analyze lease-up differences across markets:

    {market_stats}

    Explain which market performs better and why.
    """

    st.write(generate_ai_insight(prompt))

# Cluster Distribution
if "cluster" in filtered_df.columns:
    st.subheader("Cluster Distribution")

    cluster_counts = (
        filtered_df["cluster"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .reset_index()
    )
    cluster_counts.columns = ["cluster", "count"]

    fig4 = px.bar(
        cluster_counts,
        x="cluster",
        y="count",
        text="count",
        title="Cluster Size Distribution"
    )
    fig4.update_layout(
        xaxis_title="Cluster",
        yaxis_title="Number of Properties"
    )
    st.plotly_chart(fig4, use_container_width=True)


# PCA Cluster Projection
if all(col in filtered_df.columns for col in ["cluster", "PC1", "PC2"]):
    st.subheader("PCA Projection of Lease-Up Clusters")

    pca_plot_df = filtered_df.dropna(subset=["cluster", "PC1", "PC2"]).copy()
    pca_plot_df["cluster"] = pca_plot_df["cluster"].astype(int).astype(str)

    fig_pca = px.scatter(
        pca_plot_df,
        x="PC1",
        y="PC2",
        color="cluster",
        hover_data=["PropertyID", "Name", "MarketName", "LeaseUpMonths"],
        title="2D PCA Projection of Lease-Up Performance Clusters"
    )

    fig_pca.update_layout(
        xaxis_title="Principal Component 1",
        yaxis_title="Principal Component 2",
        legend_title="Cluster"
    )

    st.plotly_chart(fig_pca, use_container_width=True)


if st.button("Generate Insight for Clusters"):
    
    from openai import OpenAI
    client = OpenAI()

    summary_text = f"""
    Market: {selected_market}
    Number of properties: {len(filtered_df)}
    Avg Lease-Up Months: {filtered_df['LeaseUpMonths'].mean():.2f}
    Negative Rent Growth Share: {filtered_df['NegativeEffRentGrowth'].mean():.2%}
    Avg Initial Occupancy: {filtered_df['InitialOccupancy'].mean():.2f}
    Avg Occupancy Growth: {filtered_df['AvgMonthlyOccGrowth'].mean():.2f}
    Cluster Distribution: {filtered_df['cluster'].value_counts().to_dict()}
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a real estate data analyst. Give concise, insightful analysis."},
            {"role": "user", "content": summary_text}
        ]
    )

    st.session_state["ai_result"] = response.choices[0].message.content

if "ai_result" in st.session_state:
    st.markdown("### 📊 AI Analysis Summary")
    st.write(st.session_state["ai_result"])

# Cluster Interpretation
if "cluster" in filtered_df.columns:
    st.subheader("Cluster Interpretation")

    st.markdown("""
The clustering results reveal three distinct lease-up patterns:

- **Cluster 0: Slow & Unstable Lease-Up**  
  Lower initial occupancy, slower absorption, and relatively higher volatility.

- **Cluster 1: Moderate but Volatile Lease-Up**  
  Medium occupancy starting point, but more fluctuation during lease-up.

- **Cluster 2: Fast & Stable Lease-Up**  
  Higher initial occupancy, faster stabilization, and lower volatility.

These clusters help summarize how properties differ in lease-up behavior and market performance.
""")

# Overall Summary
if st.button("Generate Overall Summary"):
    prompt = f"""
    Summarize overall findings:

    Total properties: {len(filtered_df)}
    Avg Lease-Up: {filtered_df['LeaseUpMonths'].mean():.2f}
    Negative Rent Share: {filtered_df['NegativeEffRentGrowth'].mean():.2%}

    Provide a concise executive summary.
    """

    st.write(generate_ai_insight(prompt))
