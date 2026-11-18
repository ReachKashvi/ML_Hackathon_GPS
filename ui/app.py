"""
PathSnap — Map-Matching Evaluator UI
Main Streamlit application running on the evaluator laptop (Laptop 2).
Communicates with the FastAPI backend on Laptop 1 (or runs in local Mock mode).
"""

from __future__ import annotations

import datetime
import os
import time
from typing import Any, Dict, List, Optional
import yaml
import streamlit as st
from streamlit_folium import st_folium

from api_client import APIClient, MockAPIClient, APIError
from map_builder import build_prediction_map


# ---------------------------------------------------------
# Configuration Loader
# ---------------------------------------------------------
def load_config() -> Dict[str, Any]:
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {
        "api_url": "http://localhost:8000",
        "request_timeout_sec": 60,
        "map_defaults": {"zoom_start": 14, "tiles": "OpenStreetMap"},
        "colors": {
            "gps_points": "#2E7DFF",
            "predicted_route": "#E24B4A",
            "road_network": "#999999",
        },
    }


# ---------------------------------------------------------
# Session State Initialization & Helpers
# ---------------------------------------------------------
def init_session_state(config: Dict[str, Any]):
    if "session_log" not in st.session_state:
        st.session_state.session_log = []
    if "connected" not in st.session_state:
        st.session_state.connected = False
    if "connected_model" not in st.session_state:
        st.session_state.connected_model = ""
    if "trips" not in st.session_state:
        st.session_state.trips = []
    if "last_prediction" not in st.session_state:
        st.session_state.last_prediction = None
    if "last_raw_resp" not in st.session_state:
        st.session_state.last_raw_resp = None
    if "api_client" not in st.session_state:
        st.session_state.api_client = None
    if "mode" not in st.session_state:
        st.session_state.mode = "Live API"


def log_api_call(endpoint: str, status: Any, duration_sec: float):
    st.session_state.session_log.append({
        "timestamp": datetime.datetime.now().strftime("%H:%M:%S"),
        "endpoint": endpoint,
        "status": status,
        "duration_ms": f"{round(duration_sec * 1000, 1)} ms",
    })


# ---------------------------------------------------------
# Main App Layout
# ---------------------------------------------------------
def main():
    st.set_page_config(
        page_title="PathSnap — Map-Matching Evaluator",
        page_icon="🗺️",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    config = load_config()
    init_session_state(config)

    # HEADER
    st.title("PathSnap — Map-Matching Evaluator")
    st.caption("UE24CS352A Hackathon | Independent Evaluator Interface")

    # SIDEBAR
    with st.sidebar:
        st.header("⚙️ Connection Settings")

        # Mode Selection: Live API vs Mock Mode
        mode = st.radio(
            "Mode:",
            options=["Live API", "Mock"],
            index=0 if st.session_state.mode == "Live API" else 1,
            help="Select 'Mock' for standalone testing without Laptop 1 backend; 'Live API' for real network evaluation.",
        )

        if mode != st.session_state.mode:
            st.session_state.mode = mode
            st.session_state.connected = False
            st.session_state.connected_model = ""
            st.session_state.trips = []
            st.session_state.last_prediction = None
            st.session_state.last_raw_resp = None
            st.rerun()

        if st.session_state.mode == "Mock":
            st.warning("⚠️ Running in MOCK MODE — Local synthetic & test dataset simulation without active backend.")

        # API URL Input
        default_url = config.get("api_url", "http://localhost:8000")
        api_url = st.text_input(
            "API URL (Laptop 1):",
            value=default_url,
            help="Enter http://<laptop1_ip>:8000 for the two-laptop evaluation setup.",
        )

        timeout_sec = config.get("request_timeout_sec", 60)

        # Connect Button
        if st.button("🔗 Connect", use_container_width=True, type="primary"):
            t0 = time.time()
            try:
                if st.session_state.mode == "Mock":
                    client = MockAPIClient(base_url=api_url, timeout=timeout_sec)
                else:
                    client = APIClient(base_url=api_url, timeout=timeout_sec)

                health_data, raw_resp = client.health()
                duration = time.time() - t0

                st.session_state.api_client = client
                st.session_state.connected = True
                st.session_state.connected_model = health_data.get("model", "Unknown Model")
                log_api_call("/health", raw_resp.status_code, duration)

                # Automatically load trips after connecting
                t_trips0 = time.time()
                trips, trips_resp = client.list_trips()
                st.session_state.trips = trips
                log_api_call("/trips", trips_resp.status_code, time.time() - t_trips0)

            except APIError as e:
                duration = time.time() - t0
                st.session_state.connected = False
                st.session_state.connected_model = ""
                log_api_call("/health", e.status_code or "FAILED", duration)
                st.error(f"❌ Connection failed: {str(e)}")
            except Exception as e:
                duration = time.time() - t0
                st.session_state.connected = False
                st.session_state.connected_model = ""
                log_api_call("/health", "ERROR", duration)
                st.error(f"❌ Unexpected error: {str(e)}")

        # Connection Status Display
        if st.session_state.connected:
            st.success(f"✓ Connected to API")
            st.markdown(f"**Model:** `{st.session_state.connected_model}`")
        else:
            st.info("⚪ Not connected. Click **Connect** above.")

        st.divider()
        st.markdown(
            """
            **Evaluation Rules:**
            - Ground truth is strictly preserved server-side.
            - Metrics are calculated by Laptop 1.
            - Evaluator inspects raw API request/response below.
            """
        )

    # MAIN PANE
    if not st.session_state.connected:
        st.info("👈 Please enter the API URL and click **Connect** in the sidebar to begin evaluation.")
        st.markdown(
            """
            ### Evaluation Setup Instructions
            1. Ensure **Laptop 1** is running the FastAPI server on port `8000`:
               ```bash
               uvicorn app:app --host 0.0.0.0 --port 8000
               ```
            2. Ensure both laptops are on the same Wi-Fi / LAN network.
            3. Enter Laptop 1's local IP address into the sidebar (e.g., `http://192.168.1.42:8000`).
            4. Click **Connect** to verify connectivity via `/health`.
            5. *Optional:* Switch to **Mock Mode** in the sidebar to preview and inspect without a live backend.
            """
        )
        _render_session_log()
        return

    # Section 1: Available evaluation trips
    st.subheader("1. Available Evaluation Trips")
    col_trip_sel, col_trip_btn = st.columns([4, 1])

    with col_trip_btn:
        st.write("")  # Alignment spacing
        if st.button("🔄 Refresh Trips", use_container_width=True):
            t0 = time.time()
            try:
                trips, resp = st.session_state.api_client.list_trips()
                st.session_state.trips = trips
                log_api_call("/trips", resp.status_code, time.time() - t0)
                st.toast("Trip list refreshed!", icon="✅")
            except APIError as e:
                log_api_call("/trips", e.status_code or "FAILED", time.time() - t0)
                st.error(f"Failed to fetch trips: {str(e)}")

    with col_trip_sel:
        trips_options = st.session_state.trips if st.session_state.trips else ["No trips available"]
        selected_trip = st.selectbox(
            "Select a trip to evaluate:",
            options=trips_options,
            index=0 if trips_options else None,
            help="Select one of the 10 official student evaluation trips.",
        )

    # Section 2: Run prediction
    st.subheader("2. Run Route Prediction")
    can_predict = bool(st.session_state.trips and selected_trip and selected_trip != "No trips available")
    
    if st.button("🚀 Run Prediction", disabled=not can_predict, type="primary"):
        t0 = time.time()
        with st.spinner(f"Calling `/predict/{selected_trip}` on backend..."):
            try:
                pred_data, raw_resp = st.session_state.api_client.predict(selected_trip)
                duration = time.time() - t0
                st.session_state.last_prediction = pred_data
                st.session_state.last_raw_resp = raw_resp
                log_api_call(f"/predict/{selected_trip}", raw_resp.status_code, duration)
                st.success(f"Prediction received for trip `{selected_trip}`!")
            except APIError as e:
                duration = time.time() - t0
                log_api_call(f"/predict/{selected_trip}", e.status_code or "FAILED", duration)
                st.error(f"Prediction failed ({e.status_code}): {e.body or str(e)}")
            except Exception as e:
                duration = time.time() - t0
                log_api_call(f"/predict/{selected_trip}", "ERROR", duration)
                st.error(f"Unexpected prediction error: {str(e)}")

    # Section 3: Results section (shown after successful prediction)
    if st.session_state.last_prediction:
        pred = st.session_state.last_prediction
        metrics = pred.get("metrics", {})
        trip_id = pred.get("trip_id", selected_trip)

        st.divider()
        st.subheader(f"3. Evaluation Results — Trip `{trip_id}`")

        # Four metric cards
        top1 = metrics.get("top1_accuracy", 0.0)
        continuity = metrics.get("route_continuity", 0.0)
        n_gps = metrics.get("n_gps_points", len(pred.get("gps_points", [])))
        inf_time = metrics.get("inference_time_sec", 0.0)

        top1_display = f"{top1 * 100:.2f}%" if isinstance(top1, (int, float)) and 0 <= top1 <= 1 else f"{top1}"
        continuity_display = f"{continuity * 100:.2f}%" if isinstance(continuity, (int, float)) and 0 <= continuity <= 1 else f"{continuity}"
        inf_display = f"{inf_time:.2f} s" if isinstance(inf_time, (int, float)) else f"{inf_time}"

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("🎯 Top-1 Accuracy", top1_display)
        m_col2.metric("🔗 Route Continuity", continuity_display)
        m_col3.metric("📍 GPS Points", n_gps)
        m_col4.metric("⚡ Inference Time", inf_display)

        # Folium Map
        st.markdown("#### Interactive Geographical Map")
        map_defaults = config.get("map_defaults", {})
        colors = config.get("colors", {})

        folium_map = build_prediction_map(
            api_response=pred,
            colors=colors,
            zoom_start=map_defaults.get("zoom_start", 14),
            tiles=map_defaults.get("tiles", "OpenStreetMap"),
        )

        st_folium(
            folium_map,
            height=600,
            use_container_width=True,
            returned_objects=[],
        )

        # Expander: API Request / Response log
        with st.expander("🔍 API Request / Response Log (Evaluator Inspection)", expanded=False):
            raw_resp = st.session_state.last_raw_resp
            if raw_resp is not None:
                duration_val = getattr(raw_resp, "elapsed", None)
                duration_str = f"{duration_val.total_seconds() * 1000:.1f} ms" if duration_val else "N/A"

                info_col1, info_col2, info_col3 = st.columns(3)
                info_col1.markdown(f"**Endpoint URL:** `{raw_resp.url}`")
                info_col2.markdown(f"**HTTP Status:** `{raw_resp.status_code}`")
                info_col3.markdown(f"**Latency:** `{duration_str}`")

                st.markdown("**Full Server Response Payload (JSON):**")
                st.json(pred)
            else:
                st.json(pred)

    # Section 4: Session log expander at bottom
    _render_session_log()


def _render_session_log():
    with st.expander("📋 Session Call Log", expanded=False):
        if st.session_state.get("session_log"):
            st.dataframe(st.session_state.session_log, use_container_width=True)
        else:
            st.caption("No API calls recorded in this session yet.")


if __name__ == "__main__":
    main()
