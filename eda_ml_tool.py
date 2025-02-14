import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import time


## Set up the app 

# Streamlit Page Config
st.set_page_config(page_title="EDA and ML Tool", layout="wide")
st.title("📊EDA and ML Tool")

## CSS
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
    }
    .fade-in {
        animation: fadeIn 0.5s;
    }
    .fade-out {
        animation: fadeOut 0.5s;
    }
    @keyframes fadeIn {
        from {opacity: 0;}
        to {opacity: 1;}
    }
    @keyframes fadeOut {
        from {opacity: 1;}
        to {opacity: 0;}
    }
    </style>
    """, unsafe_allow_html=True)


## File Uploading 
upload_data = st.sidebar.file_uploader("Choose a csv File", type=['csv'])
# upload_data = 'homeprices_m.csv'

## Missing Values

def missing_values(data_df):
    
    # Missing Values
    st.write('**Null Values**')
    null_values = data_df.isnull().sum().reset_index()
    null_values.columns = ["Column", "Missing Values"]
    st.table(null_values)


    # Showing Null value heatmap
    if st.button("Click for Null Value Heatmap"):
        st.subheader("🔥 Null Value Heatmap")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(data_df.isnull(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)


    missing_values = data_df.isnull().sum().sum()    
     #handling with missing values
    if missing_values == 0:
        st.success("🎉 No missing values found in the dataset!")
    else:
        st.warning(f"⚠️ Your dataset has {missing_values} missing values.")
        option = st.selectbox(
        "How do you want to handle missing values?",
        ("Do nothing", "Drop rows with missing values", "Drop columns with missing values","Fill with Mean","Fill with Median","Fill with Custom Value"),
    )

        if option == "Drop rows with missing values":
            # data_df.dropna(inplace=True)
            st.session_state["data_df"] = data_df.dropna()
            st.write("### Data After Dropping Rows")
        elif option == "Drop columns with missing values":
            data_df.dropna(axis=1,inplace=True)
            st.write("### Data After Dropping Columns")
        elif option == "Fill with Mean":
            data_df.fillna(data_df.mean(numeric_only=True), inplace=True)
            st.write("### ✅ Data After Filling Missing Values with Mean")

        elif option == "Fill with Median":
            data_df.fillna(data_df.median(numeric_only=True), inplace=True)
            st.write("### ✅ Data After Filling Missing Values with Median")

        elif option == "Fill with Custom Value":
            custom_value = st.text_input("Enter custom value to fill missing data:")
            if custom_value:
                data_df.fillna(custom_value, inplace=True)
                st.write(f"### ✅ Data After Filling Missing Values with '{custom_value}'")
    
    st.write("✅ Final DataSet")
    st.write(st.session_state['data_df'].shape)
    st.dataframe(st.session_state["data_df"])

## Make tabs
data_overview, data_clean, data_vis, model_build, = st.tabs(["Data Overview", "Data Cleaning", "Data Visualization", "Model Building"])

if upload_data is not None:
    data_df = pd.read_csv(upload_data)

    with data_overview:
        try:
            st.sidebar.write('Data Preview')   
            st.sidebar.write(data_df.head())

            # if st.button("Start Processing"):
            if True:
                with st.spinner("Processing... Please wait ⏳"):
                    time.sleep(1)  # Simulate a delay
                st.success("Processing Complete! ✅")

                #         # Data set overview

                num_variables = data_df.shape[1]
                number_of_rows = data_df.shape[0]
                number_of_columns = data_df.shape[1]
                missing_cells = data_df.isnull().sum().sum()
                missing_percentage = (missing_cells / (num_variables * number_of_rows)) * 100
                duplicate_rows = data_df.duplicated().sum()
                duplicate_percentage = (duplicate_rows / number_of_rows) * 100
                columns = list(data_df.columns)
                num_numeric = data_df.select_dtypes(include=["number"]).shape[1]
                num_categorical = data_df.select_dtypes(include=["object", "category"]).shape[1]


                stats_dict = {
                    "Metric": [
                    "Number of Variables",
                    "Number of Rows",
                    "Number of Columns",
                    "Missing cells",
                    "Missing cells (%)",
                    "Duplicate rows",
                    "Duplicate rows (%)",
                    "columns",
                    "Numeric variables",
                    "Categorical variables",
                    ],
                    "Value": [
                        num_variables,
                        number_of_rows,
                        number_of_columns,
                        missing_cells,
                        f"{missing_percentage:.1f}%",
                        duplicate_rows,
                        f"{duplicate_percentage:.1f}%",
                        columns,
                        num_numeric,
                        num_categorical                      
                    ]
                }

                df_stats = pd.DataFrame(stats_dict)

                # Display dataset statistics in Streamlit without the index
                st.subheader("📊 Dataset Overview")
                st.table(df_stats)  # st.table() hides the index by default

                # Create 2 columns
                col1, col2, col3 = st.columns(3)

                # Display Data Types in First Column
                with col1:
                    st.subheader("🚨 Missing Values")
                    st.write(data_df.isnull().sum())
                   

                # Display Missing Values in Second Column
                with col2:
                    st.subheader("📝 Data Types")
                    st.write(data_df.dtypes)
                

                
                st.subheader('📌 Summary Stats') 
                first_rows, last_rows, summary_stats, corr_matrix, data_dist = st.tabs(["First Rows", "Last Rows", "Summary Statistics", "Correlation Matrix",
                                                                                       "Data Distribution"])
                with first_rows:
                    st.write(data_df.head())
               
                with last_rows:
                    st.write(data_df.tail())
                
                with summary_stats:
                    st.write("### 📌 Summary Statistics")
                    st.write(data_df.describe(include="all"))
                
                with corr_matrix:

                    col1, col2 = st.columns(2)

                    
                    # corr_mat, corr_map = st.tabs(["Correlation Matrix", "Correlation Heatmap"])
                    
                    with col1:
                        correlation_matrix = data_df.select_dtypes(include=['number']).corr() 
                         # Display the Correlation Matrix
                        st.write("### 🔍 Correlation Matrix (Numerical Features)")
                        st.dataframe(correlation_matrix)                      
                    with col2:
                        fig, ax = plt.subplots(figsize=(6, 6))
                        sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
                        st.pyplot(fig)


                #### DATA DISTRIBUTION
                        
                with data_dist:

                    st.subheader("Data Distribution by Count")

                    select_col = st.selectbox("Select X-axis", data_df.columns, key='data_dist')

                    if select_col:
                        fig = px.bar(data_df, x = select_col, color=select_col)
                        st.plotly_chart(fig)


        except Exception as e:
            st.write(f'Error reading {e}')
    
   
   
    with data_clean:
        try:
            st.subheader("Data Cleaning 💹")
             # Initialize session state for multiple sections
            sections = ["Missing Values", "Duplicate Rows"]

            for section in sections:
                if section not in st.session_state:
                    st.session_state[section] = False

            # Function to toggle visibility
            def toggle_section(section):
                st.session_state[section] = not st.session_state[section]

            # Create multiple toggle buttons
            for section in sections:
                if st.button(f"{section}"):
                    toggle_section(section)

                # Placeholder for animated content
                placeholder = st.empty()

                if st.session_state[section]:
                    with placeholder.container():
                        st.markdown('<div class="fade-in">', unsafe_allow_html=True)
                        
                        
                        if section == "Missing Values":
                            missing_values(data_df)
                        if section == "Duplicate Rows":
                            st.write(f"Duplicate rows {data_df.duplicated().sum()}")
                            st.write(f"{data_df[data_df.duplicated()]}")


                else:
                    with placeholder.container():
                        st.markdown('<div class="fade-out"></div>', unsafe_allow_html=True)
                    time.sleep(0.5)  # Delay for animation before hiding
                    placeholder.empty()  # Hide block after fade-out
            

        except Exception as e:
            st.write(f'Error reading {e}')

    with data_vis:
        try:
            
            st.subheader("Visualization📈📊📉")
            st.dataframe(data_df)
            st.write(data_df.shape)
            x_col = st.selectbox("Select X-axis", data_df.columns, key='data_vis_x')
            y_col = st.selectbox("Select Y-axis", data_df.columns, key='data_vis_y')


            st.write("**Select Plot Types:**")
            # Scatter Plot


            choice = st.radio("Select an option:", ["Line Plot","Scatter Plot", "Pair Plot", "Bar Plot", 
                                                    "Histogram","Pie","Funnel", "Heatmap"], horizontal=True)
            
            if choice == "Line Plot":
                st.subheader("Line Plot")
                
                fig = px.line(data_df, x=x_col, y=y_col, title=f"{y_col} Trend")
                st.plotly_chart(fig)

            if choice == "Scatter Plot":
                st.subheader("Scatter Plot")

                fig = px.scatter(data_df, x=x_col, y=y_col, color=y_col)
                st.plotly_chart(fig)

            
            if choice == "Pair Plot":
                st.subheader("Pair Plot")

                fig = sns.pairplot(data_df)
                st.pyplot(fig)

            
            if choice == "Bar Plot":
                st.subheader("Bar Plot")

                fig = px.bar(data_df, x=x_col, y=y_col, color=y_col)
                st.plotly_chart(fig)


            if choice == "Histogram":
                st.subheader("Histogram")
                st.write("**Select on X-axis")

                fig  = px.histogram(data_df, x=x_col, nbins=5,color=x_col)
                st.plotly_chart(fig)
            
            if choice == "Pie":
                st.subheader("Pie Plot")
                st.write("**Select on X-axis")
                
                fig = px.pie(data_df, names=x_col, hole=0.2)
                st.plotly_chart(fig)
            
            if choice == "Funnel":
                st.subheader("Funnel Plot")

                fig = px.funnel(data_df, x=x_col, y=y_col, color=y_col)
                st.plotly_chart(fig)
            
            if choice == "Heatmap":
                st.subheader("HeatMap")
                correlation_matrix = data_df.select_dtypes(include=['number']).corr()
                fig, ax = plt.subplots(figsize=(4,2))
                sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt='.2f', ax=ax)
                st.pyplot(fig)
        

        except Exception as e:
            st.write(f'Error reading {e}')
    
    
    with model_build:
        try:
            
            
            st.write("Model Building")

        except Exception as e:
            st.write(f'Error reading {e}')
else:
    st.warning('Upload a dataset')


