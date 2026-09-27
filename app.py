import streamlit as st
import pandas as pd
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
# -------------------------------------------------
# PAGE CONFIGURATION
# -------------------------------------------------

st.set_page_config(
    page_title="Online Shopper Segmentation",
    page_icon="🛒",
    layout="wide"
)

# -------------------------------------------------
# TITLE
# -------------------------------------------------

st.title("🛒 Online Shopper Segmentation")

st.subheader(
    "Segmenting Online Shoppers: A Combined Cluster and "
    "Logistic Regression Approach for Forecasting Purchase Behaviour"
)

st.write(
    "This dashboard presents shopper segmentation using K-Means "
    "clustering and purchase behaviour prediction using Logistic Regression."
)

st.divider()

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------

df = pd.read_csv(
    Path(__file__).resolve().parent / "online_shoppers_clustered.csv"
)
# -------------------------------------------------
# TRAIN LOGISTIC REGRESSION MODEL
# -------------------------------------------------

@st.cache_resource
def train_model():

    data = pd.read_csv(
        Path(__file__).resolve().parent / "online_shoppers_intention.csv"
    )

    # Convert boolean columns to integers
    data["Weekend"] = data["Weekend"].astype(int)
    data["Revenue"] = data["Revenue"].astype(int)

    # One-hot encode categorical columns
    data = pd.get_dummies(
        data,
        columns=["Month", "VisitorType"],
        drop_first=True,
        dtype=int
    )

    # Separate input and target
    X = data.drop("Revenue", axis=1)
    y = data["Revenue"]

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    # Scale the features
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)

    # Train Logistic Regression
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    return model, scaler, X.columns

# -------------------------------------------------
# DATASET SUMMARY
# -------------------------------------------------

st.header("📊 Dataset Summary")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Shoppers",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Total Features",
        len(df.columns) - 1
    )

with col3:
    st.metric(
        "Number of Clusters",
        df["Cluster"].nunique()
    )

st.divider()

# -------------------------------------------------
# CLUSTER SIZE
# -------------------------------------------------

st.header("👥 Shopper Distribution by Cluster")

cluster_counts = (
    df["Cluster"]
    .value_counts()
    .sort_index()
)

st.bar_chart(cluster_counts)

st.divider()

# -------------------------------------------------
# REVENUE RATE
# -------------------------------------------------

st.header("💰 Revenue Rate by Cluster")

revenue_rate = (
    df.groupby("Cluster")["Revenue"]
    .mean() * 100
)

revenue_rate = revenue_rate.round(2)

st.bar_chart(revenue_rate)

st.divider()

# -------------------------------------------------
# CLUSTER PROFILE
# -------------------------------------------------

st.header("🧩 Cluster Profile")

cluster_profile = df.groupby("Cluster").agg({
    "Administrative": "mean",
    "Informational": "mean",
    "ProductRelated": "mean",
    "ProductRelated_Duration": "mean",
    "BounceRates": "mean",
    "ExitRates": "mean",
    "PageValues": "mean",
    "Revenue": "mean"
}).round(3)

st.dataframe(
    cluster_profile,
    use_container_width=True
)

st.divider()

# -------------------------------------------------
# MODEL PERFORMANCE
# -------------------------------------------------

st.header("🤖 Model Performance")

performance = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "K-Means + Logistic Regression"
    ],
    "Accuracy (%)": [
        85.00,
        84.87
    ],
    "Precision (%)": [
        51.07,
        50.80
    ],
    "Recall (%)": [
        74.87,
        74.87
    ],
    "F1 Score (%)": [
        60.72,
        60.53
    ]
})

st.dataframe(
    performance,
    use_container_width=True
)

st.bar_chart(
    performance.set_index("Model")
)

st.divider()

# -------------------------------------------------
# FOOTER
# -------------------------------------------------

st.success(
    "Dashboard loaded successfully! "
    "K-Means clustering and Logistic Regression results are displayed."
)


# -------------------------------------------------
# PURCHASE PREDICTION
# -------------------------------------------------

st.divider()

st.header("🛒 Purchase Prediction")

st.write(
    "Enter the shopper's browsing information below "
    "to predict whether the shopper is likely to make a purchase."
)

col1, col2 = st.columns(2)

with col1:
    administrative = st.number_input(
        "Administrative Pages",
        min_value=0.0,
        value=1.0
    )

    administrative_duration = st.number_input(
        "Administrative Duration",
        min_value=0.0,
        value=50.0
    )

    informational = st.number_input(
        "Informational Pages",
        min_value=0.0,
        value=0.0
    )

    informational_duration = st.number_input(
        "Informational Duration",
        min_value=0.0,
        value=0.0
    )

    product_related = st.number_input(
        "Product Related Pages",
        min_value=0.0,
        value=20.0
    )

    product_duration = st.number_input(
        "Product Related Duration",
        min_value=0.0,
        value=500.0
    )

    bounce_rate = st.number_input(
        "Bounce Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.02
    )

with col2:
    exit_rate = st.number_input(
        "Exit Rate",
        min_value=0.0,
        max_value=1.0,
        value=0.03
    )

    page_values = st.number_input(
        "Page Values",
        min_value=0.0,
        value=5.0
    )

    special_day = st.number_input(
        "Special Day",
        min_value=0.0,
        max_value=1.0,
        value=0.0
    )

    operating_system = st.number_input(
        "Operating System",
        min_value=1,
        value=2
    )

    browser = st.number_input(
        "Browser",
        min_value=1,
        value=2
    )

    region = st.number_input(
        "Region",
        min_value=1,
        value=1
    )

    traffic_type = st.number_input(
        "Traffic Type",
        min_value=1,
        value=2
    )

    weekend = st.selectbox(
        "Weekend",
        ["No", "Yes"]
    )

    month = st.selectbox(
        "Month",
        [
            "Jan", "Feb", "Mar", "May", "June",
            "Jul", "Sep", "Oct", "Nov", "Dec"
        ]
    )

    visitor_type = st.selectbox(
        "Visitor Type",
        ["Returning Visitor", "New Visitor", "Other"]
    )
if st.button("🔮 Predict Purchase", width="stretch"):

    model, scaler, feature_columns = train_model()

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=feature_columns
    )

    input_data["Administrative"] = administrative
    input_data["Administrative_Duration"] = administrative_duration
    input_data["Informational"] = informational
    input_data["Informational_Duration"] = informational_duration
    input_data["ProductRelated"] = product_related
    input_data["ProductRelated_Duration"] = product_duration
    input_data["BounceRates"] = bounce_rate
    input_data["ExitRates"] = exit_rate
    input_data["PageValues"] = page_values
    input_data["SpecialDay"] = special_day
    input_data["OperatingSystems"] = operating_system
    input_data["Browser"] = browser
    input_data["Region"] = region
    input_data["TrafficType"] = traffic_type
    input_data["Weekend"] = 1 if weekend == "Yes" else 0

    month_column = "Month_" + month

    if month_column in input_data.columns:
        input_data[month_column] = 1

    if visitor_type == "Returning Visitor":
        input_data["VisitorType_Returning_Visitor"] = 1

    elif visitor_type == "Other":
        input_data["VisitorType_Other"] = 1

    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    st.subheader("🔮 Prediction Result")

    st.metric(
        "Purchase Probability",
        f"{probability * 100:.2f}%"
    )

    st.progress(float(probability))

    if prediction == 1:
        st.success(
        "🛒 Likely to Purchase"
        )
    else:
        st.warning(
        "❌ Unlikely to Purchase"
        )