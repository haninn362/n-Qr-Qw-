import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO, StringIO

# ----------------------
# Config
# ----------------------
st.set_page_config(
    page_title="My Streamlit App",
    page_icon="🧩",
    layout="wide",
)

# Keep small pieces of state across interactions
if "dataset" not in st.session_state:
    st.session_state.dataset = None
if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

# ----------------------
# Helper functions
# ----------------------
@st.cache_data(show_spinner=False)
def expensive_compute(x: int) -> int:
    """Example cached function you can replace with your logic."""
    s = 0
    for i in range(10_0000):
        s += (i % (x + 1))
    return s

def load_table(uploaded_file) -> pd.DataFrame:
    """Load CSV or Excel into a DataFrame."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded_file)
    elif name.endswith(".xlsx"):
        # requires openpyxl in requirements
        return pd.read_excel(uploaded_file, engine="openpyxl")
    else:
        raise ValueError("Unsupported file type. Please upload .csv or .xlsx")

# ----------------------
# Sidebar Navigation
# ----------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📤 Upload Data", "🧠 Run Logic", "📈 Explore", "⚙️ Settings"],
    index=0
)

st.sidebar.markdown("---")
with st.sidebar.expander("About this app"):
    st.write("Template to help you port Python code into a Streamlit UI and upload Excel/CSV files.")
    st.caption("Tips: use @st.cache_data for pure functions, @st.cache_resource for models/clients.")

# ----------------------
# Pages
# ----------------------
if page == "🏠 Home":
    st.title("Streamlit App Starter")
    st.write(
        """
        This is a minimal template. Drop your existing functions into the **Run Logic** page.
        Use the **Upload Data** page to load a **CSV or Excel (.xlsx)** file.
        """
    )
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Quick Demo")
        x = st.slider("Pick an integer", 1, 10, 3)
        if st.button("Compute something"):
            with st.spinner("Crunching…"):
                out = expensive_compute(x)
            st.success(f"Result: {out}")
    with col2:
        st.subheader("Session State")
        st.json({"has_dataset": st.session_state.dataset is not None,
                 "dataset_name": st.session_state.dataset_name})

elif page == "📤 Upload Data":
    st.title("Upload Data (CSV or Excel)")
    up = st.file_uploader("Upload a file", type=["csv", "xlsx"])
    if up is not None:
        try:
            df = load_table(up)
            st.session_state.dataset = df
            st.session_state.dataset_name = up.name
            st.success(f"Loaded **{up.name}** → session_state.dataset")
            st.dataframe(df.head(50))
            st.caption(f"Rows: {df.shape[0]} • Columns: {df.shape[1]}")
        except Exception as e:
            st.error(f"Failed to read file: {e}")

elif page == "🧠 Run Logic":
    st.title("Run Your Logic")
    st.write(
        """
        Replace the placeholder with your own function calls.
        Example: if you have a function `process(df, a, b)`, call it below and show the results.
        """
    )
    # Example inputs
    colA, colB = st.columns(2)
    with colA:
        a = st.number_input("Parameter A", value=0.5, step=0.1)
    with colB:
        b = st.text_input("Parameter B", value="hello")

    # Use dataset if present
    df = st.session_state.dataset
    st.caption("Dataset loaded: " + ("✅ yes" if df is not None else "❌ no"))
    if df is not None:
        st.caption(f"Using: {st.session_state.dataset_name}")

    run = st.button("Run")
    if run:
        with st.spinner("Running your logic…"):
            # --- PLACEHOLDER: swap this block with your code ---
            result = {
                "param_a": a,
                "param_b": b,
                "rows_in_dataset": int(df.shape[0]) if df is not None else 0,
                "columns": list(df.columns) if df is not None else [],
            }
            # ---------------------------------------------------
        st.success("Done!")
        st.json(result)

elif page == "📈 Explore":
    st.title("Explore Data")
    df = st.session_state.dataset
    if df is None:
        st.info("Upload a dataset on the **Upload Data** page first.")
    else:
        st.subheader("Quick summary")
        st.write("Basic `.describe()` of numeric columns:")
        st.dataframe(df.describe(include=[np.number]))
        st.subheader("Column preview")
        col_to_view = st.selectbox("Pick a column", options=df.columns)
        st.dataframe(df[[col_to_view]].head(200))

elif page == "⚙️ Settings":
    st.title("Settings")
    st.write("Put API keys, model choices, toggles, etc. here.")
    use_cache = st.toggle("Use cache", value=True)
    st.caption("If disabled, you can clear cache with `st.cache_data.clear()` and `st.cache_resource.clear()`.")
    if st.button("Clear cache"):
        st.cache_data.clear()
        st.cache_resource.clear()
        st.success("Cleared.")

# ----------------------
# Footer
# ----------------------
st.markdown("---")
st.caption("Built with ❤️ and Streamlit")
