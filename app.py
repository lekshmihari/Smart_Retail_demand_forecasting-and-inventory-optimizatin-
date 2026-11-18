import streamlit as st
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Project folder
BASE_DIR = Path(__file__).parent

# Page configuration
st.set_page_config(
    page_title="Smart Retail Inventory System",
    page_icon="📦",
    layout="wide"
)

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv(BASE_DIR / "retail_dataset.csv",
                       parse_dates=["date"])

data = load_data()

# Application Navigation

if "started" not in st.session_state:
    st.session_state.started = False

if st.session_state.started:
    st.sidebar.title("📦 Smart Retail System")

    page = st.sidebar.radio(
        "Navigate",
        [
            "Home",
            "Dashboard",
            "Inventory Management",
            "Demand Prediction",
            "Sales Analytics",
            "Model Performance"
        ],
        key="navigation"
    )
else:
    page = "Welcome"
# Welcome Screen

def start_app():
    st.session_state.started = True
    st.session_state.navigation = "Home"

def go_to_page(page_name):
    st.session_state.navigation = page_name

if not st.session_state.started:

    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {
    background-color: #071426;
    background-image:
        linear-gradient(rgba(56, 189, 248, 0.035) 1px, transparent 1px),
        linear-gradient(90deg, rgba(56, 189, 248, 0.035) 1px, transparent 1px),
        radial-gradient(circle at 50% 40%, #15365b, #071426 75%);
    background-size: 40px 40px, 40px 40px, cover;
    background-attachment: fixed;
}

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        display: none;
    }

    .block-container {
        padding-top: 2rem;
        max-width: 100%;
    }

    .welcome-card {
        min-height: 75vh;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 35px;
        background: rgba(8, 25, 48, 0.58);
        border: 1px solid rgba(56, 189, 248, 0.25);
        border-radius: 24px;
        backdrop-filter: blur(12px);
        box-shadow: 0 0 40px rgba(14, 165, 233, 0.12);
        width: 90%;
        margin: 20px auto;
        }
    

    .welcome-card {
    position: relative;
    isolation: isolate;
    }

.welcome-card::before,
.welcome-card::after {
    content: "";
    position: absolute;
    width: 75px;
    height: 75px;
    border-color: rgba(56, 189, 248, 0.65);
    border-style: solid;
    pointer-events: none;
}

.welcome-card::before {
    top: 35px;
    left: 8%;
    border-width: 1px 0 0 1px;
}

.welcome-card::after {
    bottom: 35px;
    right: 8%;
    border-width: 0 1px 1px 0;
}

    .welcome-label {
        color: #38bdf8;
        font-size: 17px;
        letter-spacing: 6px;
        font-weight: bold;
    }

.tech-label {
    display: inline-block;
    color: #38bdf8;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 3px;
    padding: 8px 16px;
    border: 1px solid rgba(56, 189, 248, 0.3);
    border-radius: 20px;
    background: rgba(56, 189, 248, 0.07);
    text-transform: uppercase;
    transform: translateY(-18px);
}
    
    .welcome-title {
    color: #ffffff !important;
    font-size: 76px;
    font-weight: 800;
    text-align: center;
    letter-spacing: 3px;
    }

    .welcome-project {
    color: #38bdf8 !important;
    font-size: 28px;
    font-weight: 600;
    margin-top: 15px;
    }

    
    .welcome-description {
    color: #bfdbfe !important;
    font-size: 17px;
    max-width: 750px;
    line-height: 1.8;
    margin-top: 15px;
    }

    .welcome-tagline {
        color: #93c5fd;
        font-size: 16px;
        margin-top: 25px;
    }
    
    div.stButton > button {
    background: linear-gradient(90deg, #2563eb, #0ea5e9);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 15px 35px;
    font-size: 18px;
    font-weight: bold;
    box-shadow: 0 0 15px rgba(14, 165, 233, 0.4);
    transition: 0.3s ease;
}

    div.stButton > button {
    background: linear-gradient(135deg, #2563eb, #0ea5e9, #38bdf8);
    color: #ffffff;
    border: 1px solid rgba(125, 211, 252, 0.5);
    border-radius: 14px;
    padding: 16px 42px;
    font-size: 17px;
    font-weight: 700;
    letter-spacing: 1px;
    box-shadow: none;
    transition: all 0.3s ease;
}

    div.stButton > button:hover {
        background: #38bdf8;
        color: #071426;
        box-shadow: none;
        transform: translateY(-2px);
    }
    
    </style>

<div class="welcome-card">
 <div class="tech-label">AI POWERED • DEMAND ANALYTICS</div>

<div class="welcome-label">MACHINE LEARNING • RETAIL ANALYTICS</div>   
    <h1 class="welcome-title">WELCOME</h1>
    <div class="welcome-project">Smart Retail Demand Forecasting</div>
    <div class="welcome-description">
        Demand Forecasting and Inventory Optimization
        Using Machine Learning.
    </div>
    <div class="welcome-tagline">Smarter Predictions. Better Decisions.</div>
</div>

    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 1, 1])

    with col2:
        st.button(
            "Get Started →",
            on_click=start_app,
            use_container_width=True
        )

    st.stop()

st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background-color: #071426;
    color: #F1F5F9;
}

[data-testid="stHeader"] {
    background: transparent;
}   

[data-testid="stSidebar"],
[data-testid="stSidebar"] section,
[data-testid="stSidebar"] > div {
    background-color: #0B1D33 !important;
}

[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
    background-color: #0B1D33 !important;
}

[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: #F1F5F9 !important;
}

[data-testid="stSidebar"] h1 {
    color: #FFFFFF !important;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"] {
    background-color: #123B68 !important;
    border-radius: 10px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-selected="true"] p {
    color: #38BDF8 !important;
    font-weight: 700;
}

.block-container {
    padding-top: 2rem;
}

h1, h2, h3 {
    color: #FFFFFF !important;
    font-weight: 700;
}

h1 {
    letter-spacing: 0.5px;
}

h2 {
    color: #BAE6FD !important;
}

h3 {
    color: #38BDF8 !important;
}

[data-testid="stMarkdownContainer"] p,
[data-testid="stMarkdownContainer"] li {
    color: #CBD5E1;
    line-height: 1.7;
}

[data-testid="stMarkdownContainer"] h1,
[data-testid="stMarkdownContainer"] h2,
[data-testid="stMarkdownContainer"] h3 {
    color: #F1F5F9 !important;
}    

[data-testid="stVerticalBlock"] > div:has(> [data-testid="stElementContainer"]) {
    background: #102944 !important;
    border: 7px solid rgba(56, 189, 248, 0.65) !important;
    border-radius: 22px !important;
    padding: 20px !important;
    box-shadow: 0 4px 18px rgba(0, 0, 0, 0.25);
}

[data-testid="stVerticalBlockBorderWrapper"]:hover {
    border-color: #38BDF8 !important;
    box-shadow: 0 0 18px rgba(56, 189, 248, 0.15);
}

div.stButton > button {
    background-color: #123B68;
    color: #F1F5F9;
    border: 1px solid #38BDF8;
    border-radius: 10px;
    font-weight: 600;
    transition: all 0.2s ease;
}

div.stButton > button:hover {
    background-color: #1E4D7A;
    color: #FFFFFF;
    border-color: #38BDF8;
}

[data-testid="stDownloadButton"] button {
    background-color: #123B68 !important;
    color: #F1F5F9 !important;
    border: 1px solid #38BDF8 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}

[data-testid="stDownloadButton"] button:hover {
    background-color: #1E4D7A !important;
    color: #FFFFFF !important;
    border-color: #38BDF8 !important;
}

[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}
[data-testid="stMetricValue"] {
    color: #FFFFFF !important;
}

[data-testid="stMetricLabel"] {
    color: #BAE6FD !important;
}



/* Dropdown control */
div[data-baseweb="select"] > div,
div[data-baseweb="select"] > div > div {
    background-color: #102944 !important;
    border: 1px solid #38BDF8 !important;
    border-radius: 10px !important;
}

div[data-baseweb="select"] span,
div[data-baseweb="select"] input,
div[data-baseweb="select"] svg {
    color: #F1F5F9 !important;
    fill: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
}

div[data-baseweb="select"] input[role="combobox"] {
    background-color: #102944 !important;
    color: #F1F5F9 !important;
    -webkit-text-fill-color: #F1F5F9 !important;
    caret-color: #F1F5F9 !important;
    border: none !important;
    box-shadow: none !important;
}

/* Dropdown options panel */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
div[data-baseweb="menu"] {
    background-color: #102944 !important;
}

div[data-baseweb="menu"] li,
div[role="option"],
li[role="option"] {
    background-color: #102944 !important;
    color: #F1F5F9 !important;
}

div[data-baseweb="menu"] li:hover,
div[role="option"]:hover,
li[role="option"]:hover {
    background-color: #123B68 !important;
    color: #FFFFFF !important;
}

[data-testid="stSelectbox"] label,
[data-testid="stSelectbox"] label p {
    color: #F1F5F9 !important;
}

</style>
""", unsafe_allow_html=True)

if page == "Home":

    st.divider()

    st.title("Welcome to Smart Retail")

    st.subheader(
        "Smarter Predictions. Better Decisions."
    )

    st.write(
        "An intelligent retail analytics platform designed "
        "to forecast product demand, monitor inventory, "
        "and support data-driven business decisions."
    )

    st.divider()

    st.subheader("Explore Your Workspace")

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.markdown("### 🔮 Demand Forecasting")
            st.write("Predict future product demand using machine learning.")
            st.button(
                "Explore Demand Forecasting →",
                on_click=go_to_page,
                args=("Demand Prediction",),
                use_container_width=True
            )
            

    with col2:
        with st.container(border=True):
            st.markdown("### 📦 Inventory Optimization")
            st.write("Monitor stock levels and identify products requiring reorder.")
            st.button(
                "Explore Inventory Management →",
                on_click=go_to_page,
                args=("Inventory Management",),
                use_container_width=True
            )

    with col3:
        with st.container(border=True):
            st.markdown("### 📊 Sales Analytics")
            st.write("Explore sales patterns and understand business performance.")
            st.button(
                "Explore Sales Analytics →",
                on_click=go_to_page,
                args=("Sales Analytics",),
                use_container_width=True
            )

# Load inventory and slow-moving data

@st.cache_data
def load_inventory():
    return pd.read_csv(BASE_DIR / "inventory_recommendations.csv")

inventory = load_inventory()

@st.cache_data
def load_slow_moving():
    return pd.read_csv(BASE_DIR / "slow_moving_products.csv")

slow_moving = load_slow_moving()

# Calculate overview metrics

reorder_products = (
    inventory["stock_status"] == "Reorder Now"
).sum()

slow_products = (
    slow_moving["product_status"] == "Slow Moving"
).sum()

if page == "Dashboard":
    # Project Overview Dashboard
    st.divider()
    st.header("📊 Project Overview Dashboard")

    # Calculate overview values
    total_sales = data["sales"].sum()
    total_products = data["item_id"].nunique()
    total_stores = data["store_id"].nunique()
    average_daily_demand = data["sales"].mean()

    # Display dashboard metrics
    col1, col2, col3 = st.columns(3)

    col1.metric("Total Units Sold", f"{total_sales:,.0f}")
    col2.metric("Total Products", total_products)
    col3.metric("Total Stores", total_stores)

    col4, col5, col6 = st.columns(3)

    col4.metric("Average Daily Demand", f"{average_daily_demand:.2f}")
    col5.metric("Products Requiring Reorder", int(reorder_products))
    col6.metric("Slow-Moving Products", int(slow_products))


    # Key Business Insights

    st.divider()
    st.header("💡 Key Business Insights")

    # Best-selling product
    best_selling_product = (
        data.groupby("item_id")["sales"]
        .sum()
        .idxmax()
    )

    # Highest-selling store
    best_store = (
        data.groupby("store_id")["sales"]
        .sum()
        .idxmax()
    )

    # Average sales with and without promotion
    no_promo = data[data["promotion"] == 0]["sales"].mean()
    promo = data[data["promotion"] == 1]["sales"].mean()

    st.markdown(
    f"- 🏆 Best-selling Product: Product {best_selling_product}\n"
    f"- 🏬 Highest-performing Store: Store {best_store}\n"
    f"- 📉 Average sales without promotion: {no_promo:.2f} units\n"
    f"- 📈 Average sales with promotion: {promo:.2f} units\n"
    f"- 📦 Products requiring reorder: {int(reorder_products)}\n"
    f"- 🐢 Slow-moving products: {int(slow_products)}"
)

    st.divider()

    # Dataset overview
    st.header("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", f"{len(data):,}")

    with col2:
        st.metric("Total Stores", data["store_id"].nunique())

    with col3:
        st.metric("Total Products", data["item_id"].nunique())

    st.subheader("Dataset Preview")

    styled_data = data.head(10).style.set_properties(
        **{
            "background-color": "#102944",
            "color": "#F1F5F9",
            "border-color": "#24415E"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#123B68"),
                ("color", "#F1F5F9"),
                ("border", "1px solid #24415E")
            ]
        }
    ])

    st.table(styled_data)

    st.subheader("Sales Summary")

    sales_summary = data["sales"].describe().to_frame(name="Sales")

    styled_summary = sales_summary.style.set_properties(
        **{
            "background-color": "#102944",
            "color": "#F1F5F9",
            "border-color": "#24415E"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#123B68"),
                ("color", "#F1F5F9"),
                ("border", "1px solid #24415E")
            ]
        }
    ])

    st.table(styled_summary)
          

if page == "Inventory Management":

    # Inventory Recommendations
    st.divider()
    st.header("📦 Inventory Recommendations")

    # Inventory summary
    reorder_count = (inventory["stock_status"] == "Reorder Now").sum()
    sufficient_count = (inventory["stock_status"] == "Stock Sufficient").sum()
    total_reorder = inventory["reorder_quantity"].sum()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Products Requiring Reorder", int(reorder_count))

    with col2:
        st.metric("Stock Sufficient", int(sufficient_count))

    with col3:
        st.metric("Total Reorder Quantity", int(total_reorder))

    # Display recommendations
    st.subheader("Inventory Details")
    styled_inventory = inventory.style.set_properties(
        **{
            "background-color": "#102944",
            "color": "#F1F5F9",
            "border-color": "#24415E"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#123B68"),
                ("color", "#F1F5F9"),
                ("border", "1px solid #24415E")
            ]
        }
    ])

    st.table(styled_inventory)

    # Download Inventory Report
    st.subheader("📥 Download Inventory Report")

    inventory_csv = inventory.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Inventory Report (CSV)",
        data=inventory_csv,
        file_name="inventory_recommendations.csv",
        mime="text/csv",
        key="download_inventory"
    )
    # Filter products
    st.subheader("Filter Inventory Status")

    status = st.selectbox(
        "Select Stock Status",
        ["All", "Reorder Now", "Stock Sufficient"],
        key="inventory_status"
    )

    if status == "All":
        filtered_inventory = inventory
    else:
        filtered_inventory = inventory[
            inventory["stock_status"] == status
        ]

    styled_filtered_inventory = filtered_inventory.style.set_properties(
    **{
        "background-color": "#102944",
        "color": "#F1F5F9",
        "border-color": "#24415E"
    }
).set_table_styles([
    {
        "selector": "th",
        "props": [
            ("background-color", "#123B68"),
            ("color", "#F1F5F9"),
            ("border", "1px solid #24415E")
        ]
    }
])

    st.table(styled_filtered_inventory)

    # Slow Moving Product Analysis
    st.divider()
    st.header("🐢 Slow-Moving Product Analysis")

    # Summary
    slow_count = (
        slow_moving["product_status"] == "Slow Moving"
    ).sum()

    normal_count = (
        slow_moving["product_status"] == "Normal Moving"
    ).sum()

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Slow-Moving Products", int(slow_count))

    with col2:
        st.metric("Normal-Moving Products", int(normal_count))

    # Display details
    st.subheader("Product Movement Details")

    styled_slow_moving = slow_moving.style.set_properties(
        **{
            "background-color": "#102944",
            "color": "#F1F5F9",
            "border-color": "#24415E"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#123B68"),
                ("color", "#F1F5F9"),
                ("border", "1px solid #24415E")
            ]
        }
    ])

    st.table(styled_slow_moving)

    # Download Slow-Moving Products Report

    st.subheader("📥 Download Slow-Moving Products Report")

    slow_moving_csv = slow_moving.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Slow-Moving Products Report (CSV)",
        data=slow_moving_csv,
        file_name="slow_moving_products.csv",
        mime="text/csv",
        key="download_slow_moving"
    )

    # Filter
    movement = st.selectbox(
        "Filter Product Movement",
        ["All", "Slow Moving", "Normal Moving"],
        key="product_movement"
    )

    if movement == "All":
        filtered_products = slow_moving
    else:
        filtered_products = slow_moving[
            slow_moving["product_status"] == movement
        ]

    styled_filtered_products = filtered_products.style.set_properties(
        **{
            "background-color": "#102944",
            "color": "#F1F5F9",
            "border-color": "#24415E"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#123B68"),
                ("color", "#F1F5F9"),
                ("border", "1px solid #24415E")
            ]
        }
    ])

    st.table(styled_filtered_products)

if page == "Model Performance":

    # Model Performance Analysis
    st.divider()
    st.header("🤖 Machine Learning Model Performance")

    @st.cache_data
    def load_comparison():
        return pd.read_csv(BASE_DIR / "model_comparison.csv")

    comparison = load_comparison()

    st.subheader("Model Comparison Table")

    styled_comparison = comparison.style.set_properties(
        **{
            "background-color": "#102944",
            "color": "#F1F5F9",
            "border-color": "#24415E"
        }
    ).set_table_styles([
        {
            "selector": "th",
            "props": [
                ("background-color", "#123B68"),
                ("color", "#F1F5F9"),
                ("border", "1px solid #24415E")
            ]
        }
    ])

    st.table(styled_comparison)
    st.subheader("Performance Comparison")

    # Clean column names
    comparison.columns = comparison.columns.str.strip()

    # Find the WAPE column
    wape_column = next(
        (col for col in comparison.columns if "wape" in col.lower()),
        None
    )

    # Rename WAPE column if necessary
    if wape_column and wape_column != "WAPE":
        comparison = comparison.rename(columns={wape_column: "WAPE"})

    metric = st.selectbox(
        "Choose Evaluation Metric",
        ["MAE", "RMSE", "WAPE"],
        key="model_metric"
    )
    if metric in comparison.columns:

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.barplot(
            data=comparison,
            x="Model",
            y=metric,
            ax=ax
        )

        ax.set_title(f"Model Comparison - {metric}")
        ax.set_ylabel(metric)
        ax.set_xlabel("Machine Learning Model")

        st.pyplot(fig)
        plt.close(fig)

    else:
        st.error(f"Column {metric} was not found.")
        st.write("Available columns:", comparison.columns.tolist())

if page == "Demand Prediction":

    # Actual Demand Prediction
    st.divider()
    st.header("🔮 Actual Demand Prediction")

    # Load trained XGBoost model
    @st.cache_resource
    def load_model():
        return joblib.load(BASE_DIR / "xgb_model.pkl")

    xgb_model = load_model()

    # Load original model feature names
    with open(BASE_DIR / "features.json", "r") as file:
        feature_columns = json.load(file)

    st.success("XGBoost model loaded successfully!")

    st.write(
        "The trained machine learning model is ready "
        "to predict retail product demand."
    )


    # Demand Prediction Form

    st.subheader("Enter Product Details")

    store_options = sorted(data["store_id"].unique())
    item_options = sorted(data["item_id"].unique())

    col1, col2 = st.columns(2)

    with col1:
        selected_store = st.selectbox(
            "Select Store",
            store_options,
            key="prediction_store"
        )

    with col2:
        selected_item = st.selectbox(
            "Select Product",
            item_options,
            key="prediction_item"
        )

    # Get selected product's history
    history = data[
        (data["store_id"] == selected_store) &
        (data["item_id"] == selected_item)
    ].sort_values("date")

    latest_record = history.iloc[-1]

    col3, col4 = st.columns(2)

    with col3:
        price = st.number_input(
            "Selling Price",
            min_value=0.01,
            value=float(latest_record["price"]),
            step=1.0
        )

    with col4:
        promotion = st.selectbox(
            "Promotion Active?",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            key="prediction_promotion"
        )

    # Predict demand
    if st.button("Predict Demand", key="predict_demand"):

        if len(history) < 7:
            st.error("Not enough sales history for prediction.")

        else:
            latest_date = history["date"].max()
            prediction_date = latest_date + pd.Timedelta(days=1)

            # Calculate recent sales features
            lag_1 = history.iloc[-1]["sales"]
            lag_7 = history.iloc[-7]["sales"]
            rolling_mean_7 = history.tail(7)["sales"].mean()

            # Prepare model input
            input_data = pd.DataFrame(
                0.0,
                index=[0],
                columns=feature_columns,
                dtype=float
            )

            input_values = {
                "price": price,
                "promotion": promotion,
                "supplier_lead_time": latest_record["supplier_lead_time"],
                "month": prediction_date.month,
                "day_of_week": prediction_date.dayofweek,
                "lag_1": lag_1,
                "lag_7": lag_7,
                "rolling_mean_7": rolling_mean_7
            }

            for feature, value in input_values.items():
                if feature in input_data.columns:
                    input_data.loc[0, feature] = value

            # Set store and product dummy variables
            store_column = f"store_id_{selected_store}"
            item_column = f"item_id_{selected_item}"

            if store_column in input_data.columns:
                input_data.loc[0, store_column] = 1

            if item_column in input_data.columns:
                input_data.loc[0, item_column] = 1

            # Generate prediction
            prediction = xgb_model.predict(input_data)[0]
            prediction = max(0, prediction)

            st.success("Demand prediction completed!")

            st.metric(
                "Predicted Product Demand",
                f"{prediction:.2f} units"
            )

            st.write("Prediction Date:", prediction_date.date())
            st.write("Selected Store:", selected_store)
            st.write("Selected Product:", selected_item)
            # Demand Forecast Chart
            st.subheader("Recent Sales vs Predicted Demand")

            chart_data = history.tail(7)[["date", "sales"]].copy()
            chart_data = chart_data.rename(columns={"sales": "Demand"})

            predicted_row = pd.DataFrame({
                "date": [prediction_date],
                "Demand": [prediction]
            })

            chart_data = pd.concat(
                [chart_data, predicted_row],
                ignore_index=True
            )
            fig, ax = plt.subplots(figsize=(10, 4))

            ax.plot(
                chart_data["date"][:-1],
                chart_data["Demand"][:-1],
                marker="o",
                label="Actual Sales"
            )

            ax.scatter(
                chart_data["date"].iloc[-1],
                chart_data["Demand"].iloc[-1],
                color="red",
                marker="*",
                s=150,
                label="Predicted Demand"
            )

            ax.set_title("Last 7 Days and Next-Day Demand Prediction")
            ax.set_xlabel("Date")
            ax.set_ylabel("Units Sold")
            ax.legend()
            ax.grid(True)

            plt.xticks(rotation=45)
            st.pyplot(fig)
            plt.close(fig)
if page == "Sales Analytics":     

    # Product-wise Sales Analysis

    with st.expander("📊 Product-wise Sales Analysis", expanded=False):

        product_options = sorted(data["item_id"].unique())

        selected_product_analysis = st.selectbox(
            "Select a Product to Analyse",
            product_options,
            key="product_sales_analysis"
        )

        product_data = data[
            data["item_id"] == selected_product_analysis
        ]

        daily_product_sales = product_data.groupby(
            "date"
        )["sales"].sum().reset_index()

        # Product Sales Summary Metrics

        total_sales = daily_product_sales["sales"].sum()
        average_sales = daily_product_sales["sales"].mean()
        highest_sales = daily_product_sales["sales"].max()

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Units Sold", f"{total_sales:.0f}")
        col2.metric("Average Daily Sales", f"{average_sales:.2f}")
        col3.metric("Highest Daily Sales", f"{highest_sales:.0f}")

        st.subheader(f"Sales Trend for Product {selected_product_analysis}")

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            daily_product_sales["date"],
            daily_product_sales["sales"],
            marker="o",
            label="Daily Sales"
        )

        ax.set_title("Product-wise Daily Sales Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Units Sold")
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)

    # Store-wise Sales Analysis

    st.divider()


    with st.expander("🏬 Store-wise Sales Analysis"):

        store_options = sorted(data["store_id"].unique())

        selected_store_analysis = st.selectbox(
            "Select a Store to Analyse",
            store_options,
            key="store_sales_analysis"
        )

        store_data = data[
            data["store_id"] == selected_store_analysis
        ]

        # Calculate daily sales
        daily_store_sales = store_data.groupby(
            "date"
        )["sales"].sum().reset_index()

        # Calculate store performance
        total_store_sales = daily_store_sales["sales"].sum()
        average_store_sales = daily_store_sales["sales"].mean()
        highest_store_sales = daily_store_sales["sales"].max()

        # Find best-selling product
        best_product = (
            store_data.groupby("item_id")["sales"]
            .sum()
            .idxmax()
        )

        # Display summary metrics
        col1, col2, col3 = st.columns(3)

        col1.metric("Total Store Sales", f"{total_store_sales:.0f}")
        col2.metric("Average Daily Sales", f"{average_store_sales:.2f}")
        col3.metric("Highest Daily Sales", f"{highest_store_sales:.0f}")

        st.info(f"Best-selling Product: {best_product}")

        # Display sales trend
        st.subheader(f"Sales Trend for Store {selected_store_analysis}")

        fig, ax = plt.subplots(figsize=(12, 4))

        ax.plot(
            daily_store_sales["date"],
            daily_store_sales["sales"],
            marker="o",
            label="Daily Store Sales"
        )

        ax.set_title("Store-wise Daily Sales Trend")
        ax.set_xlabel("Date")
        ax.set_ylabel("Total Units Sold")
        ax.legend()
        ax.grid(True)

        plt.xticks(rotation=45)
        st.pyplot(fig)
        plt.close(fig)


    # Promotion Impact Analysis

    st.divider()
    st.header("🎯 Promotion Impact Analysis")

    promotion_sales = data.groupby("promotion")["sales"].mean().reset_index()

    promotion_sales["Promotion Status"] = promotion_sales["promotion"].map({
        0: "No Promotion",
        1: "Promotion Active"
    })

    col1, col2 = st.columns(2)

    no_promotion_sales = promotion_sales.loc[
        promotion_sales["promotion"] == 0, "sales"
    ].iloc[0]

    active_promotion_sales = promotion_sales.loc[
        promotion_sales["promotion"] == 1, "sales"
    ].iloc[0]

    col1.metric("Average Sales Without Promotion", f"{no_promotion_sales:.2f}")
    col2.metric("Average Sales With Promotion", f"{active_promotion_sales:.2f}")

    # Calculate percentage change
    percentage_change = (
        (active_promotion_sales - no_promotion_sales)
        / no_promotion_sales
    ) * 100

    st.metric("Promotion Sales Impact", f"{percentage_change:.2f}%")

    # Promotion comparison chart
    fig, ax = plt.subplots(figsize=(8, 4))

    sns.barplot(
        data=promotion_sales,
        x="Promotion Status",
        y="sales",
        ax=ax
    )

    ax.set_title("Average Sales: Promotion vs No Promotion")
    ax.set_xlabel("Promotion Status")
    ax.set_ylabel("Average Units Sold")

    st.pyplot(fig)
    plt.close(fig)

    st.divider()
    st.header("📊 Additional Business Visualizations")

    # Daily Sales Trend Overview

    st.subheader("📈 Overall Daily Sales Trend")

    daily_sales = data.groupby("date")["sales"].sum().reset_index()

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(
        daily_sales["date"],
        daily_sales["sales"],
        linewidth=2,
        label="Daily Sales"
    )

    ax.set_title("Daily Retail Sales Trend")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Units Sold")
    ax.legend()
    ax.grid(True)

    plt.xticks(rotation=45)
    st.pyplot(fig)
    plt.close(fig)
