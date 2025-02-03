import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns


## Set up the app 

# Streamlit Page Config
st.set_page_config(page_title="EDA and ML Tool", layout="wide")
st.title("📊EDA and ML Tool")


## File Uploading 
upload_data = st.sidebar.file_uploader("Choose a csv File", type=['csv'])
upload_data = 'homeprices_m.csv'


if upload_data:
    data_df = pd.read_csv(upload_data)
    missing_values = data_df.isnull().sum().sum()
    eda_selected = st.sidebar.checkbox("📈 Show EDA", value=False)
    dataviz = st.sidebar.checkbox("📈 Data Visualization", value=False)
    
    if not eda_selected:
        st.write('File Uploaded')
        # Data set overview
        st.write("**Overview**")
        st.write(f"🔢 **Shape**: {data_df.shape}")
        st.write(f'**Columns:** {list(data_df.columns)}')
        if missing_values > 0:
            st.warning(f"⚠️ Your dataset has {missing_values} missing values.")
        
                
    if eda_selected:      
        # showing data
        st.subheader("Uploaded Data Preview")
        st.write(data_df)
        
        # Data set overview
        st.subheader("📌 Dataset Overview")
        st.write(f"🔢 Number of Rows: {data_df.shape[0]}")
        st.write(f"🔠 Number of Columns: {data_df.shape[1]}")
        st.write(f'**Columns:** {list(data_df.columns)}')
        st.write("**Data types of each Column**")
        st.write(data_df.dtypes)

        # Missing Values
        st.subheader('🚨 Missing Values')
        st.write(data_df.isnull().sum())
        

        
        #handling with missing values
        if missing_values == 0:
            st.success("🎉 No missing values found in the dataset!")
        else:
            st.warning(f"⚠️ Your dataset has {missing_values} missing values.")
            option = st.selectbox(
            "How do you want to handle missing values?",
            ("Do nothing", "Drop rows with missing values", "Drop columns with missing values"),
        )

            if option == "Drop rows with missing values":
                data_df = data_df.dropna()
                st.write("### Data After Dropping Rows")
            elif option == "Drop columns with missing values":
                data_df = data_df.dropna(axis=1)
                st.write("### Data After Dropping Columns")
            elif option == "Fill with Mean":
                data_df = data_df.fillna(data_df.mean())
                st.write("### ✅ Data After Filling Missing Values with Mean")

            elif option == "Fill with Median":
                data_df = data_df.fillna(data_df.median())
                st.write("### ✅ Data After Filling Missing Values with Median")

            elif option == "Fill with Custom Value":
                custom_value = st.text_input("Enter custom value to fill missing data:")
                if custom_value:
                    data_df = data_df.fillna(custom_value)
                    st.write(f"### ✅ Data After Filling Missing Values with '{custom_value}'")

        st.write("✅ Final DataSet")
        st.write(data_df.shape)
        st.dataframe(data_df)


        # Summary statistics
        st.subheader("📌 Summary Statistics")
        st.write(data_df.describe(include="all"))

        correlation_matrix = data_df.corr()
    
        # Display the Correlation Matrix
        st.write("### 🔍 Correlation Matrix (Numerical Features)")
        st.dataframe(correlation_matrix)



    # --- Handling Categorical Variables ---

    #later

    # --- Data Visualization ---
    if dataviz:

        st.subheader("📊 Data Visualization")

         # Scatter Plot
        if st.sidebar.checkbox("Scatter Plot"):
            st.subheader("📌 Scatter Plot")
            plt.subplots(figsize=(10, 6))
            x_col = st.selectbox("Select X-axis", data_df.columns)
            y_col = st.selectbox("Select Y-axis", data_df.columns)
            sns.scatterplot(x=x_col, y=y_col, data=data_df)
            st.pyplot(plt)
        
        #histogram
        if st.sidebar.checkbox("Histogram"):
            st.subheader("📈 Histogram")
            column = st.selectbox("Select a column for histogram", data_df.columns)
            if data_df[column].dtype in ["int64", "float64"]:
                plt.figure(figsize=(8, 4))
                sns.histplot(data_df[column], kde=True)
                plt.title(f"Histogram of {column}")
                st.pyplot(plt.gcf())
            else:
                st.write(f"The selected column `{column}` is not numeric.")

        # Pair Plot
        if st.sidebar.checkbox("Show Pair Plot"):
            st.subheader("📈 Pair Plot")
            fig = sns.pairplot(data_df)
            st.pyplot(fig)

        # Correlation Heatmap
        if st.sidebar.checkbox("Correlation Heatmap"):
            st.subheader("🔥 Correlation Heatmap")
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.heatmap(data_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

       

        





    
    