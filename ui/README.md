# PathSnap — Evaluator UI (Phase 6)

The evaluator-facing web interface built using **Streamlit** and **Folium** for the **UE24CS352A Hackathon**.

This application is deployed on **Laptop 2 (Evaluator Machine)** to inspect, evaluate, and visualize map-matching predictions produced by the **Laptop 1 (Inference Server)** backend.

---

## 🔒 Security & Evaluation Rules

1. **Strict Zero Ground-Truth Policy:** The UI never loads, sees, or calculates against ground-truth labels.
2. **Server-Side Validation:** All performance metrics (`top1_accuracy`, `route_continuity`, `n_gps_points`, `n_predicted_arcs`, `inference_time_sec`) are calculated server-side by Laptop 1 and transmitted via the API.
3. **Transparent Audit Log:** The UI includes visible request/response inspection panels so evaluators can verify raw HTTP traffic and timing.

---

## 📦 Prerequisites & Installation

On Laptop 2 (or your local testing machine):

```bash
# 1. Clone or navigate to the repository root
cd "PathSnap"

# 2. Install required dependencies
pip install -r requirements.txt
```

Required packages:
- `streamlit`
- `folium`
- `streamlit-folium`
- `requests`
- `pyyaml`

---

## 🚀 How to Run

From the project root:

```bash
streamlit run ui/app.py
```

The browser will automatically open to `http://localhost:8501`.

---

## 🌐 Connecting to Laptop 1 (Two-Laptop Setup)

### 1. Find Laptop 1's IP Address
On **Laptop 1**, open terminal / command prompt and check the local IPv4 address on the shared Wi-Fi / LAN network:
- **Windows:**
  ```cmd
  ipconfig
  ```
  Look for *IPv4 Address* under your active Wi-Fi adapter (e.g. `192.168.1.45`).
- **macOS / Linux:**
  ```bash
  ifconfig | grep "inet "
  # or
  ip -br addr
  ```

### 2. Configure Laptop 1 API Server
Ensure Laptop 1 is running FastAPI with host `0.0.0.0` to accept incoming connections across the LAN:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

### 3. Point the UI to Laptop 1
- Open the UI on Laptop 2 (`http://localhost:8501`).
- In the **Sidebar**, set the **API URL** to:
  ```text
  http://<laptop1_ip>:8000
  ```
  *(Example: `http://192.168.1.45:8000`)*
- Click **🔗 Connect**.
- Once connected, the sidebar displays a green `✓ Connected to API` and shows the server's model name.

---

## 🧪 Mock Mode (Offline Development & Demo)

Before the Phase 5 FastAPI backend is running or when testing standalone without network setup:
1. In the sidebar, switch the **Mode** radio button from `Live API` to `Mock`.
2. A yellow warning banner will confirm: `Running in MOCK MODE`.
3. Click **🔗 Connect**.
4. The client will automatically simulate the API contract using local data files from `student_test` (e.g., trip `00000091`) or synthetic fallback data.
5. You can test the trip selector, run predictions, examine the Folium map, and review API logs without any server running.

---

## 🗺️ Map Layers & Features

The Folium map includes an interactive layer control toggle:
- **GPS Observations (Blue Circles):** Raw GPS coordinates (`lon`, `lat`, timestamp).
- **Predicted Route (Thick Red Polylines):** The sequence of road segments matched by the model.
- **Road Network (Gray Polylines):** The underlying road network graph in the trip vicinity.
- **Start / End Markers:** Green marker at trip origin, red marker at trip destination.

---

## 📁 File Structure

```
ui/
├── app.py              # Main Streamlit web application
├── api_client.py       # API client wrapping requests + MockAPIClient
├── map_builder.py      # Interactive Folium map generator
├── config.yaml         # Default API URL, timeouts, and layer colors
└── README.md           # Documentation and run instructions
```
