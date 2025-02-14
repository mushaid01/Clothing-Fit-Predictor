import streamlit as st
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import io
# Load the model and preprocessing objects
rf_loaded = joblib.load('Model and Objects/random_forest_model.pkl')
scaler = joblib.load('Model and Objects/scaler.pkl')
label_encoders = joblib.load('Model and Objects/label_encoders.pkl')
target_encoder = joblib.load('Model and Objects/target_encoder.pkl')

# Define options for categorical fields
bust_size_options = ['34d', '34b', '34c', '32b', '36d', '36a', '32d', '36c', '32c', '36b', '34a',]
rented_for_options = ['vacation', 'formal affair', 'wedding', 'date', 'everyday', 'party', 'work']
body_type_options = ['hourglass', 'straight & narrow', 'pear', 'athletic', 'full bust', 'petite', 'apple']
category_options = ['romper', 'gown', 'dress', 'sheath', 'leggings', 'top', 'jumpsuit', 'sweater', 'jacket', 'maxi']
cup_size_options = ['d', 'b', 'c', 'a', 'f', 'g']

# Streamlit App
st.set_page_config(page_title="Clothing Fit Predictor", layout="wide")
st.title("👗 Clothing Fit Predictor")
st.write("### Fill in the details below to predict how well a clothing item will fit you.")

# Sidebar for Dashboard Navigation
st.sidebar.title("📊 Dashboard")
page = st.sidebar.radio("Go to", ("Fit Predictor", "Data Insights"))

if page == "Fit Predictor":
    col1, col2 = st.columns(2)
    with col1:
        bust_size = st.selectbox("Bust Size", bust_size_options)
        rented_for = st.selectbox("Rented For", rented_for_options)
        body_type = st.selectbox("Body Type", body_type_options)
        category = st.selectbox("Category", category_options)
        cup_size = st.selectbox("Cup Size", cup_size_options)
    
    with col2:
        rating = st.slider("Rating", min_value=1.0, max_value=10.0, step=0.1, value=8.0)
        size = st.number_input("Size", min_value=0, max_value=50, value=6)
        age = st.number_input("Age", min_value=18, max_value=80, value=30)
        weight_num = st.number_input("Weight (lbs)", min_value=80, max_value=300, value=140)
        height_in = st.number_input("Height (inches)", min_value=48, max_value=80, value=65)
    
    BMI = weight_num / ((height_in / 39.37) ** 2)
    bust_number = int(bust_size[:-1])

    if st.button("🔍 Predict Fit"):
        user_input = {
            'bust size': bust_size,
            'rating': rating,
            'rented for': rented_for,
            'body type': body_type,
            'category': category,
            'size': size,
            'age': age,
            'weight_num': weight_num,
            'height_in': height_in,
            'BMI': BMI,
            'bust_number': bust_number,
            'cup_size': cup_size
        }
        
        for col in ['bust size', 'rented for', 'body type', 'category', 'cup_size']:
            if user_input[col] in label_encoders[col].classes_:
                user_input[col] = label_encoders[col].transform([user_input[col]])[0]
            else:
                most_frequent = label_encoders[col].classes_[0]
                user_input[col] = label_encoders[col].transform([most_frequent])[0]
                st.warning(f"Unseen category in {col}, replaced with '{most_frequent}'")
        
        user_df = pd.DataFrame([user_input])
        user_scaled = scaler.transform(user_df)
        predicted_fit = rf_loaded.predict(user_scaled)
        predicted_fit_label = target_encoder.inverse_transform(predicted_fit)
        
        st.markdown(f"## 🎯 **Predicted Fit:** {predicted_fit_label[0]}")
        st.success("The prediction is based on your inputs!")

elif page == "Data Insights":
    # Google Drive File ID (Extracted from the link)
    file_id = "1TIS2jac7i7IJqL0h9wxFsqyQoFOqlM4y"

    # Google Drive Direct Download URL
    download_url = f"https://drive.google.com/uc?export=download&id={file_id}"

    # Send HTTP request to get the file
    response = requests.get(download_url)
    response.raise_for_status()  # Raise error if request fails

    # Read the CSV file into a Pandas DataFrame (without saving to disk)
    df_cleaned = pd.read_csv(io.StringIO(response.text))
    rental_counts = df_cleaned['rented for'].value_counts().reset_index()
    rental_counts.columns = ['rented for', 'count']
    fig3 = px.bar(rental_counts, x='rented for', y='count', title='Count of Rental Reasons', labels={'rented for': 'Rental Reason', 'count': 'Count'}, color='rented for', text='count')
    st.plotly_chart(fig3)
    
    grouped_bmi = df_cleaned.groupby('body type')['BMI'].mean().reset_index()
    fig4 = px.bar(grouped_bmi, x='body type', y='BMI', title='Average BMI by Body Type', labels={'body type': 'Body Type', 'BMI': 'Average BMI'}, color='body type', text='BMI')
    st.plotly_chart(fig4)
    
    top_categories = df_cleaned['category'].value_counts().head(5).index.tolist()
    filtered_df = df_cleaned[df_cleaned['category'].isin(top_categories)]
    sunburst_df_filtered = filtered_df.groupby(['category', 'rented for']).size().reset_index(name='count')
    fig6 = px.sunburst(sunburst_df_filtered, path=['category', 'rented for'], values='count', title='Sunburst: Rental Reasons by Top 5 Product Categories', color='count', color_continuous_scale='RdBu')
    st.plotly_chart(fig6)

    numeric_cols = ['age', 'rating', 'weight_num', 'height_in', 'BMI', 'size']
    corr = df_cleaned[numeric_cols].corr()

    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale='RdBu_r', title='Correlation Heatmap', labels=dict(color="Correlation"))
    st.plotly_chart(fig_corr)
    
    text = " ".join(str(review) for review in df_cleaned['review_text'].dropna())
    from wordcloud import WordCloud
    wc = WordCloud(width=800, height=400, background_color='white').generate(text)
    fig_wordcloud = px.imshow(wc.to_array())
    fig_wordcloud.update_xaxes(visible=False)
    fig_wordcloud.update_yaxes(visible=False)
    fig_wordcloud.update_layout(title="Word Cloud of Review Texts")
    st.plotly_chart(fig_wordcloud)
