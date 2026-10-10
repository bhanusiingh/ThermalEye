"""Streamlit Dashboard for ThermalEye."""
import streamlit as st
import pandas as pd
import pydeck as pdk
from pathlib import Path

# Adjust path if running directly from script
import sys
ROOT_DIR = Path(__file__).resolve().parent.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.thermaleye.dashboard.data_loader import load_events_data, filter_events

st.set_page_config(
    page_title="ThermalEye Investigation Dashboard",
    page_icon="🔥",
    layout="wide"
)

# Cache data loading so it doesn't read from disk on every widget interaction
@st.cache_data
def get_data():
    return load_events_data()

def format_val(val, default="Unavailable"):
    """Format missing numeric values for UI display."""
    if pd.isna(val):
        return default
    if isinstance(val, float):
        return round(val, 3)
    return val

def render_overview(df: pd.DataFrame):
    """Render high-level metrics and map."""
    st.header("ThermalEye India Investigation Overview")
    
    if df.empty:
        st.warning("No prioritized events found. Please run the fusion pipeline.")
        return
        
    st.markdown("""
    **Disclaimer:** This system prioritizes candidate thermal anomalies for human review based on heuristic evidence fusion and uncalibrated visual similarity (EfficientNet-B0). It **does not** output confirmed industrial fires.
    """)
    
    # Metrics
    total = len(df)
    high = len(df[df['priority'] == 'High'])
    med = len(df[df['priority'] == 'Medium'])
    low = len(df[df['priority'] == 'Low'])
    insuf = len(df[df['priority'] == 'Insufficient evidence'])
    
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Events", total)
    m2.metric("🔴 High Priority", high)
    m3.metric("🟠 Medium Priority", med)
    m4.metric("🟡 Low Priority", low)
    m5.metric("⚪ Insufficient Evidence", insuf)
    
    # Map
    st.subheader("Event Locations (India)")
    st.markdown("Displays events passing the India geographic boundary validation.")
    
    # Use PyDeck for colored scatterplot
    view_state = pdk.ViewState(
        latitude=22.5,
        longitude=79.5,
        zoom=4.2,
        pitch=0
    )
    
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=df,
        get_position=["centroid_lon", "centroid_lat"],
        get_color="map_color",
        get_radius=15000, # 15km for visibility at national scale
        pickable=True,
    )
    
    tooltip = {
        "html": "<b>Event ID:</b> {event_id} <br/>"
                "<b>Priority:</b> {priority} <br/>"
                "<b>Score:</b> {priority_score} <br/>"
                "<b>Coverage:</b> {evidence_coverage}",
        "style": {"backgroundColor": "steelblue", "color": "white"}
    }
    
    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view_state,
        tooltip=tooltip,
        map_style="mapbox://styles/mapbox/light-v9"
    )
    st.pydeck_chart(deck)

def render_investigation_panel(df: pd.DataFrame):
    """Render the detailed investigation table and filters."""
    st.header("Investigation Database")
    
    # Filters
    c1, c2, c3 = st.columns([1, 2, 2])
    search_id = c1.text_input("Search Event ID", "")
    
    all_priorities = ['High', 'Medium', 'Low', 'Insufficient evidence']
    priorities = c2.multiselect("Filter Priority", all_priorities, default=all_priorities)
    
    coverage_options = ['firms', 'temporal', 'osm', 'cv']
    coverage = c3.multiselect("Require Evidence Sources", coverage_options, default=[])
    
    filtered_df = filter_events(df, search_id, priorities, coverage)
    
    st.markdown(f"**Showing {len(filtered_df)} of {len(df)} events**")
    
    # Columns to show in the high-level table
    table_cols = ['event_id', 'priority', 'priority_score', 'evidence_coverage', 'mean_frp', 'active_days']
    
    if filtered_df.empty:
        st.info("No events match the current filters.")
        return
        
    st.dataframe(
        filtered_df[table_cols].sort_values(by="priority_score", ascending=False).style.format(na_rep="Unavailable"),
        use_container_width=True,
        hide_index=True
    )
    
    st.subheader("Event Detail Inspector")
    selected_event = st.selectbox("Select an event to inspect detailed evidence", filtered_df['event_id'].tolist())
    
    if selected_event:
        event_data = filtered_df[filtered_df['event_id'] == selected_event].iloc[0]
        
        st.markdown(f"### {selected_event} ({event_data['priority']})")
        st.markdown(f"**Total Priority Score:** {format_val(event_data['priority_score'])} (Weight basis: {format_val(event_data['available_evidence_weight'])})")
        
        ec1, ec2, ec3, ec4 = st.columns(4)
        
        with ec1:
            st.markdown("#### 🔥 FIRMS")
            st.markdown(f"**Score Component:** {format_val(event_data.get('score_firms'))}")
            st.markdown(f"- Mean FRP: {format_val(event_data.get('mean_frp'))}")
            st.markdown(f"- Max FRP: {format_val(event_data.get('max_frp'))}")
            st.markdown(f"- Mean Conf: {format_val(event_data.get('mean_firms_confidence'))}")
            st.markdown(f"- Detections: {format_val(event_data.get('detection_count'))}")
            
        with ec2:
            st.markdown("#### ⏱️ Temporal")
            st.markdown(f"**Score Component:** {format_val(event_data.get('score_temporal'))}")
            st.markdown(f"- Active Days: {format_val(event_data.get('active_days'))}")
            st.markdown(f"- Persistence: {format_val(event_data.get('persistence_score'))}")
            st.caption("Note: Single-day persistence evaluates intra-day passes, not multi-day continuous burns.")
            
        with ec3:
            st.markdown("#### 🏭 OSM Context")
            st.markdown(f"**Score Component:** {format_val(event_data.get('score_osm'))}")
            st.markdown(f"- Data Avail: {event_data.get('osm_data_available')}")
            st.markdown(f"- Ind <1km: {format_val(event_data.get('industrial_features_1km'))}")
            st.markdown(f"- Ind Dist: {format_val(event_data.get('nearest_industrial_feature_km'))} km")
            st.markdown(f"- Power Dist: {format_val(event_data.get('nearest_power_plant_km'))} km")
            if not pd.isna(event_data.get('score_osm')) and event_data.get('score_osm') == 0.0:
                st.caption("Score 0.0 indicates successful query but no mapped industrial features. It is not definitive proof of non-industry.")
            
        with ec4:
            st.markdown("#### 🛰️ Satellite CV")
            st.markdown(f"**Score Component:** {format_val(event_data.get('score_cv'))}")
            st.caption("Evaluates visual similarity to EuroSAT industrial land-cover, uncalibrated for active fires.")
            
def render_visualizations(df: pd.DataFrame):
    """Render analytical charts."""
    st.header("Analytics")
    if df.empty:
        return
        
    import altair as alt
    
    c1, c2 = st.columns(2)
    
    with c1:
        st.subheader("Priority Distribution")
        dist = df['priority'].value_counts().reset_index()
        dist.columns = ['Priority', 'Count']
        
        base1 = alt.Chart(dist).encode(
            x=alt.X('Count:Q', title='Number of Events'),
            y=alt.Y('Priority:N', sort=['High', 'Medium', 'Low', 'Insufficient evidence'], title='')
        )
        bars1 = base1.mark_bar(color='#ff4b4b')
        text1 = base1.mark_text(align='left', baseline='middle', dx=3).encode(text='Count:Q')
        st.altair_chart((bars1 + text1).properties(height=300), use_container_width=True)
        
    with c2:
        st.subheader("Evidence Coverage Patterns")
        # Map raw internal flags to readable names
        cov = df['evidence_coverage'].value_counts().reset_index()
        cov.columns = ['Coverage Pattern', 'Count']
        
        def readable_cov(raw):
            if raw == "none" or not raw:
                return "No Evidence"
            parts = raw.split("|")
            mapping = {"firms": "FIRMS", "temporal": "Temporal", "osm": "OSM", "cv": "CV"}
            return " + ".join([mapping.get(p, p) for p in parts])
            
        cov['Coverage Pattern'] = cov['Coverage Pattern'].apply(readable_cov)
        
        base2 = alt.Chart(cov).encode(
            x=alt.X('Count:Q', title='Number of Events'),
            y=alt.Y('Coverage Pattern:N', sort='-x', title='')
        )
        bars2 = base2.mark_bar(color='#1f77b4')
        text2 = base2.mark_text(align='left', baseline='middle', dx=3).encode(text='Count:Q')
        st.altair_chart((bars2 + text2).properties(height=300), use_container_width=True)

def main():
    df = get_data()
    
    tab1, tab2, tab3 = st.tabs(["🌎 Overview & Map", "🔍 Investigation Table", "📊 Analytics"])
    
    with tab1:
        render_overview(df)
        
    with tab2:
        render_investigation_panel(df)
        
    with tab3:
        render_visualizations(df)

if __name__ == "__main__":
    main()
