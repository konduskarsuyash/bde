import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql.functions import count, hour, collect_list, countDistinct, size
import sqlite3
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from pyspark.sql.functions import length, countDistinct, collect_list
from pyspark import SparkConf


# Set page config
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
    .getOrCreate()


def get_active_interactions_spark():
    conn = sqlite3.connect("users.db")
    
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

def prepare_activity_data_spark(interactions_df):
    # Group interactions by username and count the occurrences
    activity_df = interactions_df \
        .groupBy("username") \
        .agg(count("*").alias("count")) \
        .orderBy("count", ascending=False)
    return activity_df

def get_chat_history():
    conn = sqlite3.connect("users.db")
    query = "SELECT message FROM chat_history WHERE message LIKE ?"
    c = conn.cursor()
    c.execute(query, ('%?',))  # Assuming questions end with a question mark
    rows = c.fetchall()
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
    conn = sqlite3.connect("users.db")
    query = "SELECT region, COUNT(*) as Users FROM users GROUP BY region"
    region_data = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert to Spark DataFrame
    return spark.createDataFrame(region_data)

def get_user_region(user_id):
    conn = sqlite3.connect("users.db")
    c = conn.cursor()
    c.execute("SELECT region FROM users WHERE user_id = ?", (user_id,))
    result = c.fetchone()
    conn.close()
    if result:
        return result[0]
    return "Unknown"

def get_country_code(region):
    # Map city/region names to their corresponding country codes
    region_to_country = {
        'mumbai': 'IND',  # India
        'new york': 'USA', # United States
        # Add more mappings as needed
    }
    return region_to_country.get(region.lower(), region)


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
        # Create a new column with country codes
        df['country_code'] = df['region'].apply(get_country_code)

        # Create an empty figure
        fig = go.Figure()

        # Add scatter_geo for all regions (yellow dots)
        fig.add_scattergeo(
            lat=[19.0760 if r.lower() == 'mumbai' else 40.7128 if r.lower() == 'new york' else None for r in df['region']],
            lon=[72.8777 if r.lower() == 'mumbai' else -74.0060 if r.lower() == 'new york' else None for r in df['region']],
            text=df["region"],
            marker=dict(
                size=10,
                color="yellow",
                opacity=0.8,
                line=dict(width=1, color='white')
            ),
            hoverinfo="text",
            name="User Locations"
        )

        # If user is logged in, highlight their location
        if user_id and user_region != "Unknown":
            user_lat = 19.0760 if user_region.lower() == 'mumbai' else 40.7128 if user_region.lower() == 'new york' else None
            user_lon = 72.8777 if user_region.lower() == 'mumbai' else -74.0060 if user_region.lower() == 'new york' else None
            
            if user_lat and user_lon:
                fig.add_scattergeo(
                    lat=[user_lat],
                    lon=[user_lon],
                    text=[f"Your Location: {user_region}"],
                    marker=dict(
                        size=15,
                        symbol="star",
                        color="red",
                        opacity=1,
                        line=dict(width=1.5, color='white')
                    ),
                    hoverinfo="text",
                    name="Your Location"
                )

        # Update layout for dark theme and disable background color
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            geo=dict(
                showframe=False,
                showcoastlines=True,
                projection_type='equirectangular',
                bgcolor='rgba(0,0,0,0)',
                showcountries=True,
                countrycolor='gray'
            ),
            height=500
        )

        # Display the figure
        st.plotly_chart(fig, use_container_width=True)


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


    # Create a line graph for user interactions
    st.subheader("User Interactions Over Time")

    # Get the interaction data
    active_interactions_df = active_interactions_spark_df.toPandas()
    active_interactions_df['timestamp'] = pd.to_datetime(active_interactions_df['timestamp'])
    active_interactions_df = active_interactions_df.set_index('timestamp')

    # Resample the data to get hourly counts
    hourly_interactions = active_interactions_df.resample('H').size().reset_index(name='count')

    # Create the line graph with enhanced styling
    fig_line = px.line(
        hourly_interactions,
        x='timestamp',
        y='count',
        title='User Interactions Over Time',
        labels={'timestamp': 'Time', 'count': 'Number of Interactions'}
    )

    # Update layout for dark theme with better styling
    fig_line.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        xaxis_title="Time",
        yaxis_title="Number of Interactions",
        xaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)',
            tickformat='%H:%M\n%Y-%m-%d'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='rgba(255, 255, 255, 0.1)'
        ),
        showlegend=False,
        hovermode='x unified',
        # plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=30, l=10, r=10, b=10)
    )

    # Update line styling
    fig_line.update_traces(
        line_color='#00ff00',  # Bright green color for better visibility
        line_width=2,
        hovertemplate='Time: %{x}<br>Interactions: %{y}<extra></extra>'
    )

    # Display the line graph
    st.plotly_chart(fig_line, use_container_width=True)

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

