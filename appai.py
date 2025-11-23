import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit_shadcn_ui as ui 
import os
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np

# --- 1. Page Configuration (Updated with Favicon) ---
st.set_page_config(
    page_title="AI in Healthcare Ethiopia",
    page_icon="logo.png", # <--- Updated! Use "🇪🇹" or "logo.png" here
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. MINIMAL CLEAN CSS (Light Mode) ---
st.markdown("""
<style>
    /* Force White Background */
    .stApp {
        background-color: #FFFFFF;
        color: #0F172A; /* Slate 900 for text */
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #F8FAFC; /* Slate 50 */
        border-right: 1px solid #E2E8F0;
    }
    
    /* Remove extra top padding */
    .block-container {
        padding-top: 2rem;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Plotly Chart Backgrounds */
    .js-plotly-plot .plotly .main-svg {
        background-color: transparent !important;
    }
</style>
""", unsafe_allow_html=True)

# --- 3. Data Loading (SMART FIX) ---
@st.cache_data
def load_data():
    # A. Load Survey Data
    try:
        df = pd.read_csv('AI_Healthcare_Cleaned.csv')
    except FileNotFoundError:
        st.error("Critical Error: 'AI_Healthcare_Cleaned.csv' is missing.")
        st.stop()

    # Feature Engineering
    knw_cols = [c for c in df.columns if c.startswith('Knw_')]
    for c in knw_cols:
        df[c] = df[c].astype(str).str.upper().map({'TRUE': 1, 'FALSE': 0, 'NAN': 0}).fillna(0)
    df['Score_Knowledge'] = df[knw_cols].sum(axis=1)

    freq_map = {'Never': 0, 'Rarely': 1, 'Sometimes': 2, 'Often': 3, 'Always': 4}
    df['Score_Practice'] = df['Prac_Use_Validated'].map(freq_map)
    df['High_Usage'] = df['Prac_Use_Validated'].isin(['Often', 'Always'])
    df['Trust_Score'] = df['Prac_Trust_Recommend'].map(freq_map)

    likert_map = {'Strongly Disagree': 1, 'Disagree': 2, 'Neutral': 3, 'Agree': 4, 'Strongly Agree': 5}
    df['Score_Fear'] = df['Att_Replace_Doctors'].map(likert_map)
    df['Score_Optimism'] = df['Att_Benefits_Risks'].map(likert_map)
    
    df['Income'] = pd.to_numeric(df['Income'], errors='coerce')
    
    return df

df = load_data()

# --- 4. Sidebar Filters ---
with st.sidebar:
    st.header("AI Pulse")
    
    page = st.radio("Navigation", ["Dashboard", "Regional Map", "Analysis", "Policy Insights"])
    
    st.markdown("---")
    st.subheader("Global Filters")
    
    role_filter = st.multiselect("Role", df['Role'].unique())
    region_filter = st.multiselect("Region", df['Region'].unique())
    
    if role_filter:
        df = df[df['Role'].isin(role_filter)]
    if region_filter:
        df = df[df['Region'].isin(region_filter)]
        
    st.markdown("---")
    st.caption(f"Analyzing n = {len(df)}")

# --- 5. Main Content ---

if page == "Dashboard":
    st.title("AI in Healthcare Ethiopia")
    st.markdown("Overview Dashboard")
    st.markdown("---")
    
    # --- SHADCN UI METRICS ---
    cols = st.columns(4)
    
    with cols[0]:
        ui.metric_card(title="Adoption Rate", content=f"{df['High_Usage'].mean()*100:.1f}%", description="Validated Tools Usage", key="card1")
    with cols[1]:
        ui.metric_card(title="Fear Index", content=f"{df['Score_Fear'].mean():.1f}/5.0", description="Avg Anxiety Score", key="card2")
    with cols[2]:
        ui.metric_card(title="AI Literacy", content=f"{df['Score_Knowledge'].mean():.1f}/8.0", description="Avg Knowledge Score", key="card3")
    with cols[3]:
        ui.metric_card(title="Regulation", content=f"{df['Att_Regulation'].isin(['Agree', 'Strongly Agree']).mean()*100:.1f}%", description="Demand Consensus", key="card4")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Charts
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Workforce Participation")
        if not df.empty:
            top_roles = df['Role'].value_counts().nlargest(5).reset_index()
            top_roles.columns = ['Role', 'Count']
            fig = px.bar(top_roles, x='Count', y='Role', orientation='h', 
                         color='Count', color_continuous_scale='Blues')
            fig.update_layout(template="plotly_white", showlegend=False, height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No data available for selected filters.")
            
    with c2:
        st.subheader("Experience Distribution")
        if not df.empty:
            fig2 = px.histogram(df, x="Experience_Years", nbins=15, 
                                color_discrete_sequence=['#0F172A']) # Dark slate for contrast
            fig2.update_layout(template="plotly_white", height=350, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("No data available.")

elif page == "Regional Map":
    st.title("Regional Intelligence")
    
    # --- GEOPANDAS LOGIC (SMART FOLDER FIX) ---
    shapefile_path = None
    
    # Search Logic
    current_dir = os.getcwd()
    files_in_dir = os.listdir(current_dir)
    for f in files_in_dir:
        if f.endswith(".shp"):
            shapefile_path = f
            break
            
    if not shapefile_path:
        possible_folders = ["eth_admin_boundaries", "eth_admin_boundaries.shp"]
        for folder in possible_folders:
            if os.path.isdir(folder):
                subfiles = os.listdir(folder)
                for f in subfiles:
                    if f.endswith(".shp"):
                        shapefile_path = os.path.join(folder, f)
                        break
            if shapefile_path: break
            
    if shapefile_path:
        try:
            try:
                gdf = gpd.read_file(shapefile_path, layer="eth_admin1")
            except:
                gdf = gpd.read_file(shapefile_path)
            
            region_summ = df.groupby('Region').agg(
                Rate=('High_Usage', lambda x: x.mean() * 100),
                Count=('Region', 'count')
            ).reset_index()
            
            fix = {'Benishangul Gumuz': 'Benishangul Gumz', 'Gambella': 'Gambela', 
                   'Central Ethiopia': 'SNNP', 'South Ethiopia': 'SNNP', 'South West Ethiopia': 'SNNP'}
            region_summ['Map_Name'] = region_summ['Region'].replace(fix)
            
            merge_col = None
            for col in ['adm1_name', 'NAME_1', 'ADM1_EN']:
                if col in gdf.columns:
                    merge_col = col
                    break
            
            if merge_col:
                map_data = gdf.merge(region_summ, left_on=merge_col, right_on='Map_Name', how='left')
                
                col_map, col_data = st.columns([2, 1])
                
                with col_map:
                    st.write(f"**Adoption Intensity** (Source: {os.path.basename(shapefile_path)})")
                    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
                    map_data.plot(column='Rate', ax=ax, legend=True,
                                  legend_kwds={'label': "Adoption Rate (%)", 'orientation': "horizontal"},
                                  cmap='Blues', edgecolor='0.8', missing_kwds={'color': 'lightgrey', 'label': 'No Data'})
                    ax.axis('off')
                    st.pyplot(fig)
                
                with col_data:
                    st.markdown("##### Regional Data")
                    st.dataframe(
                        region_summ[['Region', 'Rate', 'Count']].sort_values('Rate', ascending=False).style.background_gradient(cmap="Blues"), 
                        use_container_width=True, 
                        height=500
                    )
            else:
                st.error(f"Shapefile loaded, but could not find Region Name column. Columns found: {list(gdf.columns)}")
                
        except Exception as e:
            st.error(f"Error loading map: {e}")
    else:
        st.warning("⚠️ Shapefile (.shp) not found in the current folder.")
        st.info("Please ensure you have uploaded the files `eth_admin_boundaries.shp`, `.shx`, and `.dbf` to this directory.")

elif page == "Analysis":
    st.title("Strategic Analysis")
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Knowledge vs Trust")
        if not df.empty:
            mkt_data = df.groupby('Role').agg(
                Knw=('Score_Knowledge', 'mean'),
                Trust=('Trust_Score', 'mean'),
                Cnt=('Role', 'count')
            ).reset_index()
            mkt_data = mkt_data[mkt_data['Cnt'] > 2]
            
            fig_bub = px.scatter(mkt_data, x="Knw", y="Trust", size="Cnt", color="Role", 
                                 size_max=60, template="plotly_white")
            fig_bub.update_layout(height=400, xaxis_title="Knowledge", yaxis_title="Trust", margin=dict(l=0,r=0,t=30,b=0))
            st.plotly_chart(fig_bub, use_container_width=True)
        
    with c2:
        st.subheader("Digital Equity Hierarchy")
        if not df.empty:
            fig_sun = px.sunburst(df, path=['Region', 'Sex', 'Prac_Use_Validated'], 
                                  color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_sun.update_layout(template="plotly_white", height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_sun, use_container_width=True)

elif page == "Policy Insights":
    st.title("Policy Pulse")
    
    col_pol1, col_pol2 = st.columns(2)
    
    with col_pol1:
        st.markdown("##### Statistical Triangulation")
        if not df.empty:
            corr = df[['Score_Knowledge', 'Score_Practice', 'Score_Fear', 'Score_Optimism']].corr()
            fig_corr = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1)
            fig_corr.update_layout(template="plotly_white", height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_corr, use_container_width=True)
    
    with col_pol2:
        st.markdown("##### Sentiment Breakdown")
        if not df.empty:
            pol_data = {
                'Metric': ['Want Regulation', 'Fear Job Loss', 'Optimistic'],
                'Value': [
                    df['Att_Regulation'].isin(['Agree', 'Strongly Agree']).mean() * 100,
                    df['Att_Replace_Doctors'].isin(['Agree', 'Strongly Agree']).mean() * 100,
                    df['Att_Benefits_Risks'].isin(['Agree', 'Strongly Agree']).mean() * 100
                ]
            }
            fig_bar = px.bar(pd.DataFrame(pol_data), x='Value', y='Metric', orientation='h', color='Metric')
            fig_bar.update_layout(template="plotly_white", showlegend=False, height=400, margin=dict(l=0,r=0,t=0,b=0))
            st.plotly_chart(fig_bar, use_container_width=True)