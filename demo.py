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

def calculate_customer_satisfaction():
    """
    Calculate customer satisfaction percentage based on chat history sentiment.
    Positive messages are assumed to reflect customer satisfaction.
    """
    conn = sqlite3.connect("users.db", check_same_thread=False)
    cursor = conn.cursor()
    
    try:
        # Query to count positive and total messages
        cursor.execute("SELECT COUNT(*) FROM chat_history WHERE role = 'positive'")
        positive_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM chat_history")
        total_count = cursor.fetchone()[0]
        
        if total_count == 0:
            # Avoid division by zero; return 0% satisfaction if no data exists
            return 0.0
        
        # Calculate satisfaction as a percentage
        satisfaction_percentage = (positive_count / total_count) * 100
        return round(satisfaction_percentage, 2)
    
    finally:
        # Ensure resources are properly released
        cursor.close()
        conn.close()



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
    
    with col_satisfaction:
        # Customer Satisfaction Pie Chart
        satisfaction = calculate_customer_satisfaction()
        fig_satisfaction = go.Figure(data=[go.Pie(
            values=[satisfaction, 100-satisfaction],
            labels=['Satisfied', 'Other'],
            hole=0.7,
            marker_colors=['#4CAF50', '#263238'],
            textinfo='percent',
            showlegend=False
        )])
        
        fig_satisfaction.update_layout(
            title="Customer Satisfaction",
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=30, l=0, r=0, b=0),
            annotations=[dict(
                text=f"{satisfaction}%",
                x=0.5, y=0.5,
                font_size=24,
                font_color='white',
                showarrow=False
            )]
        )
        
        st.plotly_chart(fig_satisfaction, use_container_width=True)
    
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