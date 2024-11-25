import sqlite3
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, hour, collect_list, countDistinct, size
from pyspark.sql.functions import length, countDistinct, collect_list
from pyspark import SparkConf
import os
import plotly.graph_objects as go
from datetime import timedelta
import requests
# Set page config
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="User Geographic Distribution",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

spark = SparkSession.builder \
    .appName("UserDistributionApp") \
    .config("spark.master", "local[*]") \
    .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow") \
    .config("spark.driver.memory", "2g") \
    .config("spark.executor.memory", "2g") \
    .getOrCreate()

def get_db_connection():
    """Create and return a database connection"""
    return sqlite3.connect("users.db")

def get_active_interactions_spark():
    conn = get_db_connection()
    
    # Use a SQL JOIN to fetch username and timestamp from active_interactions and users tables
    query = """
    SELECT active_interactions.timestamp, users.username
    FROM active_interactions
    JOIN users ON active_interactions.user_id = users.user_id
    """
    
    # Load the data into a DataFrame
    active_interactions_df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Check if the DataFrame is empty
    if active_interactions_df.empty:
        st.warning("No active interactions found in the database.")
        return spark.createDataFrame(pd.DataFrame(columns=["timestamp", "username"]))  # Return empty Spark DataFrame with schema
    
    return spark.createDataFrame(active_interactions_df)
def get_query_count():
    """Get total number of queries from chat_history"""
    conn = get_db_connection()
    cursor = conn.cursor()  # Create the cursor
    cursor.execute("SELECT COUNT(*) FROM chat_history")
    count = cursor.fetchone()[0]
    cursor.close()  # Explicitly close the cursor
    conn.close()
    return count


def get_common_topics():
    """Analyze messages to get common topics"""
    conn = get_db_connection()
    cursor = conn.cursor()  # Explicitly create the cursor
    cursor.execute("SELECT message FROM chat_history")
    messages = cursor.fetchall()
    cursor.close()  # Explicitly close the cursor
    conn.close()

    
    # Sample topics - you should implement actual message analysis here
    topics = {
        'Experience': 8,
        'Im': 10,
        'Incredible': 6,
        'Luffy': 12,
        'Madara': 12,
        'Name': 9,
        'Powerful': 9
    }
    
    return pd.DataFrame(list(topics.items()), columns=['topic', 'count'])



def get_active_interactions():
    conn = get_db_connection()
    query = """
    SELECT timestamp 
    FROM active_interactions 
    ORDER BY timestamp
    """
    active_interactions_df = pd.read_sql_query(query, conn)
    conn.close()
    
    if active_interactions_df.empty:
        st.warning("No interaction data available.")
        return pd.DataFrame(columns=['timestamp'])
    
    # Convert timestamp strings to datetime objects with a more flexible approach
    active_interactions_df['timestamp'] = pd.to_datetime(
        active_interactions_df['timestamp'],
        format='mixed',
        dayfirst=False,
        yearfirst=True
    )
    return active_interactions_df

def prepare_activity_data_spark(interactions_df):
    # Group interactions by username and count the occurrences
    activity_df = interactions_df \
        .groupBy("username") \
        .agg(count("*").alias("count")) \
        .orderBy("count", ascending=False)
    return activity_df

def get_chat_history():
    conn = get_db_connection()
    query = "SELECT message FROM chat_history WHERE message LIKE ?"
    cursor = conn.cursor()  # Explicitly create the cursor
    cursor.execute(query, ('%?%',))  # Assuming questions end with a question mark
    rows = cursor.fetchall()
    cursor.close()  # Explicitly close the cursor
    conn.close()

    return [row[0] for row in rows]

def count_most_asked_questions(questions):
    # Convert the list of questions into a pandas DataFrame
    df = pd.DataFrame(questions, columns=['question'])
    
    # Count the occurrences of each question
    question_counts = df['question'].value_counts().reset_index()
    question_counts.columns = ['question', 'count']
    
    return question_counts

def get_region_data_spark():
    # Connect to SQLite and fetch data
    conn = get_db_connection()
    query = "SELECT region, COUNT(*) as Users FROM users GROUP BY region"
    region_data = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert to Spark DataFrame
    return spark.createDataFrame(region_data)

def get_user_region(user_id):
    conn = get_db_connection()
    with conn:
        cursor = conn.cursor()
        cursor.execute("SELECT region FROM users WHERE user_id = ?", (user_id,))
        result = cursor.fetchone()
    if result:
        return result[0]
    return "Unknown"

@st.cache_data(ttl=86400)  # Cache for 24 hours
def get_lat_lon(region):
    base_url = "https://nominatim.openstreetmap.org/search"
    params = {"q": region, "format": "json"}
    headers = {
        "User-Agent": "AI Assistant/1.0 (suyashhhh123@gmail.com)"
    }
    try:
        response = requests.get(base_url, params=params, headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        if data:
            return float(data[0]["lat"]), float(data[0]["lon"])
        else:
            return 0, 0
    except Exception as e:
        st.error(f"Error fetching coordinates for {region}: {e}")
        return 0, 0
    
def prepare_map_data(df):
    # Precompute coordinates outside the map rendering
    df["coordinates"] = df["region"].apply(lambda r: get_lat_lon(r))
    df["latitude"] = df["coordinates"].apply(lambda x: x[0] if x else None)
    df["longitude"] = df["coordinates"].apply(lambda x: x[1] if x else None)
    return df

def run_dashboard():
    # Custom CSS for dark theme
    st.markdown("""
        <style>
        .stApp {
            background-color: #0E1117;
            color: #FAFAFA;
        }
        .stSelectbox {
            background-color: #262730;
        }
        </style>
    """, unsafe_allow_html=True)

    # Title
    st.title("Global User Distribution and Interaction")

    # Load user data by region
    df_spark = get_region_data_spark()
    df = df_spark.toPandas()  # Convert to pandas for visualization in Streamlit

    # Assume user_id is stored in session state after login
    user_id = st.session_state.get("user_id")

    # Retrieve and display the logged-in user's region if available
    if user_id:
        user_region = get_user_region(user_id)
        st.sidebar.header("User Information")
        st.sidebar.write(f"Region: {user_region}")

    # Main layout with columns
    col1, col2 = st.columns([2, 1])
    with col1:
        # Precompute coordinates BEFORE map rendering
        df = prepare_map_data(df)

        # Initialize a folium map centered on a default location
        m = folium.Map(
            location=[20.0, 0.0], 
            zoom_start=2,
            zoomControl=False,  # Disable zoom controls
            scrollWheelZoom=False,  # Disable scroll wheel zooming
            dragging=False  # Disable map dragging
        )

        # Add markers ONCE
        for _, row in df.iterrows():
            if row["latitude"] and row["longitude"]:
                folium.CircleMarker(
                    location=[row["latitude"], row["longitude"]],
                    radius=6,
                    color="yellow",
                    fill=True,
                    fill_opacity=0.8,
                    popup=row["region"]
                ).add_to(m)

        # Highlight logged-in user's location
        if user_id and user_region != "Unknown":
            user_lat, user_lon = get_lat_lon(user_region)
            if user_lat and user_lon:
                folium.Marker(
                    location=[user_lat, user_lon],
                    icon=folium.Icon(color="red", icon="star"),
                    popup=f"Your Location: {user_region}"
                ).add_to(m)

        # Display the map in Streamlit
        st_folium(
            m, 
            width=700,  # Fixed width
            height=500,  # Fixed height
        )    
    with col2:
        st.subheader("Users by Region")
        
        # Display the data from the database in a formatted way
        display_df = df.copy()
        display_df['Users'] = display_df['Users'].apply(lambda x: f"{x:,}")
        
        st.dataframe(
            display_df,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Key Metrics")
        total_users = df['Users'].sum()
        
        st.metric("Total Users", f"{total_users:,}")

    # Fetch and prepare activity data by username
    active_interactions_spark_df = get_active_interactions_spark()
    activity_spark_df = prepare_activity_data_spark(active_interactions_spark_df)

    # Convert Spark DataFrame to Pandas DataFrame for plotting
    activity_df = activity_spark_df.toPandas()

    # Plotting the activity data by username
    st.subheader("User Activity by Username")
    fig_activity = px.bar(
        activity_df,
        x='username',
        y='count',
        labels={'username': 'Username', 'count': 'Number of Interactions'},
        title='User Activity by Username',
        color='count',
        color_continuous_scale=px.colors.sequential.Plasma
    )

    # Update layout for dark theme
    fig_activity.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Username",
        yaxis_title="Number of Interactions",
    )

    # Display the plot
    st.plotly_chart(fig_activity, use_container_width=True)

    st.title("Global User Distribution and Interaction")
    
    # Create line graph for user interactions
    st.subheader("Trend of User Sessions")
    active_interactions_df = get_active_interactions()
    
    if not active_interactions_df.empty:
        # Filter for the past 7 days
        past_7_days = datetime.now() - timedelta(days=7)
        filtered_interactions = active_interactions_df[
            active_interactions_df['timestamp'] >= past_7_days
        ]
        
        # Group by day and count interactions
        filtered_interactions['day_of_week'] = filtered_interactions['timestamp'].dt.day_name()
        daily_interactions = (
            filtered_interactions.groupby('day_of_week')
            .size()
            .reindex(['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday'], fill_value=0)
            .reset_index(name='count')
        )
        
        # Create the line graph
        fig_line = go.Figure()
        fig_line.add_trace(
            go.Scatter(
                x=daily_interactions['day_of_week'],
                y=daily_interactions['count'],
                fill='tozeroy',
                fillcolor='rgba(65, 105, 225, 0.2)',
                line=dict(color='rgb(65, 105, 225)', width=2),
                mode='lines+markers',
                marker=dict(size=8, color='rgb(65, 105, 225)'),
                hovertemplate='%{y} sessions<extra></extra>'
            )
        )
        
        # Update layout
        fig_line.update_layout(
            title='User Sessions Over the Past Week',
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title='Day of Week',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            yaxis=dict(
                title='Number of Interactions',
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)'
            ),
            hovermode='x unified',
            showlegend=False
        )
        
        # Display the plot
        st.plotly_chart(fig_line, use_container_width=True)
    else:
        st.warning("No interaction data available for visualization.")

    # Update line styling
    fig_line.update_traces(
        line_color='#00ff00',  # Bright green color for better visibility
        line_width=2,
        hovertemplate='Time: %{x}<br>Interactions: %{y}<extra></extra>'
    )
    
        # Analytics Dashboard Section
    st.title("Text Analytics Dashboard")
    
    # Number of queries
    num_queries = get_query_count()
    st.subheader(f"Number of queries: {num_queries}")
    
    # Create two columns for charts
    col_satisfaction, col_topics = st.columns(2)
    
    
    with col_topics:
        # Common Topics Bar Chart
        topics_df = get_common_topics()
        fig_topics = go.Figure(data=[go.Bar(
            x=topics_df['count'],
            y=topics_df['topic'],
            orientation='h',
            marker_color='#64B5F6'
        )])
        
        fig_topics.update_layout(
            title="Common Topics",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(l=0, r=10, t=30, b=0),
            xaxis=dict(
                showgrid=True,
                gridcolor='rgba(255,255,255,0.1)',
                range=[0, 13]
            ),
            yaxis=dict(
                showgrid=False,
                color='white'
            ),
            font=dict(color='white')
        )
        
        st.plotly_chart(fig_topics, use_container_width=True)

        # Add download button for data
    st.download_button(
            label="Download Data",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name=f'user_distribution.csv',
            mime='text/csv'
        )

# Call run_dashboard to display the Streamlit app
if __name__ == "__main__":
    run_dashboard()