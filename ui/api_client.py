"""
PathSnap UI - API Client
Handles communication with the FastAPI map-matching backend (or local Mock mode).
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import Any, Tuple, List, Dict, Optional
import requests


class APIError(Exception):
    """Custom exception raised when API requests fail or return error status codes."""
    def __init__(self, message: str, status_code: Optional[int] = None, body: Optional[str] = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

    def __str__(self) -> str:
        base = super().__str__()
        if self.status_code:
            return f"[{self.status_code}] {base}"
        return base


class MockResponse:
    """Mock requests.Response object for MockAPIClient."""
    def __init__(self, url: str, json_data: Any, status_code: int = 200, elapsed_sec: float = 0.08):
        self.url = url
        self._json_data = json_data
        self.status_code = status_code
        self.elapsed = datetime.timedelta(seconds=elapsed_sec)
        self.text = json.dumps(json_data, indent=2)

    def json(self) -> Any:
        return self._json_data


class APIClient:
    """
    Thin wrapper around requests.Session to communicate with the PathSnap FastAPI backend.
    Each method returns (parsed_json, raw_response).
    """

    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 60.0):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> Tuple[Any, requests.Response]:
        url = f"{self.base_url}{endpoint}"
        t0 = time.time()
        # Use (connect_timeout, read_timeout): fast failure if unreachable, ample time for inference
        req_timeout = kwargs.pop("timeout", (3.0, self.timeout))
        try:
            resp = self.session.request(method, url, timeout=req_timeout, **kwargs)
        except requests.exceptions.Timeout as e:
            raise APIError(
                f"API not reachable at {self.base_url} (Timeout after {self.timeout}s). Check Laptop 1 is running the server and the IP is correct.",
                status_code=408,
                body=str(e),
            )
        except requests.exceptions.ConnectionError as e:
            raise APIError(
                f"API not reachable at {self.base_url}. Check Laptop 1 is running the server and the IP is correct.",
                status_code=503,
                body=str(e),
            )
        except requests.exceptions.RequestException as e:
            raise APIError(
                f"Request failed: {str(e)}",
                status_code=500,
                body=str(e),
            )

        if not (200 <= resp.status_code < 300):
            raise APIError(
                f"API returned HTTP {resp.status_code} for {endpoint}",
                status_code=resp.status_code,
                body=resp.text,
            )

        try:
            parsed = resp.json()
        except Exception as e:
            raise APIError(
                f"Failed to parse JSON response from {endpoint}: {resp.text[:200]}",
                status_code=resp.status_code,
                body=resp.text,
            )

        return parsed, resp

    def health(self) -> Tuple[Dict[str, Any], requests.Response]:
        """
        Check API health.
        Returns ({"status": "ok", "model": "<name>"}, response)
        """
        return self._request("GET", "/health")

    def list_trips(self) -> Tuple[List[str], requests.Response]:
        """
        List available trips for evaluation.
        Returns (list_of_trip_ids, response)
        """
        data, resp = self._request("GET", "/trips")
        if isinstance(data, dict) and "trips" in data:
            trips = data["trips"]
        elif isinstance(data, list):
            trips = data
        else:
            trips = []
        return trips, resp

    def predict(self, trip_id: str) -> Tuple[Dict[str, Any], requests.Response]:
        """
        Predict route for a given trip.
        Returns (prediction_dict, response)
        """
        return self._request("GET", f"/predict/{trip_id}")


class MockAPIClient:
    """
    Mock API client for development and demonstration before Phase 5 backend is ready.
    Loads real trip data locally if available, or generates realistic synthetic data.
    """

    AVAILABLE_TRIPS = [
        "00000000", "00000007", "00000009", "00000040", "00000052",
        "00000056", "00000065", "00000076", "00000077", "00000091"
    ]

    def __init__(self, base_url: str = "http://localhost:8000 (Mock Mode)", timeout: float = 60.0):
        self.base_url = base_url
        self.timeout = timeout
        self.connected_model = "PathSnap-HMM-Viterbi-LocalMock"

    def health(self) -> Tuple[Dict[str, Any], MockResponse]:
        data = {
            "status": "ok",
            "model": self.connected_model,
            "version": "1.0.0-mock",
            "mode": "mock"
        }
        resp = MockResponse(f"{self.base_url}/health", data, status_code=200, elapsed_sec=0.03)
        return data, resp

    def list_trips(self) -> Tuple[List[str], MockResponse]:
        data = {"trips": self.AVAILABLE_TRIPS}
        resp = MockResponse(f"{self.base_url}/trips", data, status_code=200, elapsed_sec=0.04)
        return self.AVAILABLE_TRIPS, resp

    def predict(self, trip_id: str) -> Tuple[Dict[str, Any], MockResponse]:
        t0 = time.time()
        # Attempt to load local student_test data if present
        payload = self._load_local_trip(trip_id)
        if not payload:
            payload = self._generate_synthetic_trip(trip_id)

        duration = max(0.05, round(time.time() - t0, 3))
        payload["metrics"]["inference_time_sec"] = duration
        resp = MockResponse(f"{self.base_url}/predict/{trip_id}", payload, status_code=200, elapsed_sec=duration)
        return payload, resp

    def _find_trip_dir(self, trip_id: str) -> Optional[str]:
        search_paths = [
            os.path.join("Student folder", "Student folder", "student_test", trip_id),
            os.path.join("Student folder", "student_test", trip_id),
            os.path.join("student_test", trip_id),
            os.path.join("..", "Student folder", "Student folder", "student_test", trip_id),
        ]
        for p in search_paths:
            if os.path.exists(p) and os.path.isdir(p):
                return p
        return None

    def _load_local_trip(self, trip_id: str) -> Optional[Dict[str, Any]]:
        trip_dir = self._find_trip_dir(trip_id)
        if not trip_dir:
            return None

        track_file = os.path.join(trip_dir, f"{trip_id}.track")
        nodes_file = os.path.join(trip_dir, f"{trip_id}.nodes")
        arcs_file = os.path.join(trip_dir, f"{trip_id}.arcs")
        route_file = os.path.join(trip_dir, f"{trip_id}.route")

        if not (os.path.exists(track_file) and os.path.exists(nodes_file) and os.path.exists(arcs_file)):
            return None

        # 1. Parse GPS track [lon, lat, t]
        gps_points: List[List[float]] = []
        with open(track_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    gps_points.append([float(parts[0]), float(parts[1]), float(parts[2])])

        if not gps_points:
            return None

        # 2. Parse nodes [lon, lat]
        node_coords: List[List[float]] = []
        with open(nodes_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    node_coords.append([float(parts[0]), float(parts[1])])

        # 3. Parse arcs [from_idx, to_idx]
        arc_pairs: List[List[int]] = []
        with open(arcs_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    arc_pairs.append([int(parts[0]), int(parts[1])])

        # 4. Parse route (predicted arcs)
        predicted_arcs: List[int] = []
        if os.path.exists(route_file):
            with open(route_file, "r", encoding="utf-8") as f:
                for line in f:
                    val = line.strip()
                    if val:
                        predicted_arcs.append(int(val))

        # Build predicted_arc_coords: [[[lon1, lat1], [lon2, lat2]], ...]
        predicted_arc_coords: List[List[List[float]]] = []
        for arc_id in predicted_arcs:
            if 0 <= arc_id < len(arc_pairs):
                u, v = arc_pairs[arc_id]
                if u < len(node_coords) and v < len(node_coords):
                    predicted_arc_coords.append([node_coords[u], node_coords[v]])

        # Filter road network to bounding box + buffer to keep display lightweight
        lons = [p[0] for p in gps_points]
        lats = [p[1] for p in gps_points]
        pad = 0.008
        min_lon, max_lon = min(lons) - pad, max(lons) + pad
        min_lat, max_lat = min(lats) - pad, max(lats) + pad

        # Map original node indices to filtered node indices
        filtered_node_coords: List[List[float]] = []
        node_map: Dict[int, int] = {}
        for idx, (lon, lat) in enumerate(node_coords):
            if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
                node_map[idx] = len(filtered_node_coords)
                filtered_node_coords.append([lon, lat])

        filtered_arc_pairs: List[List[int]] = []
        for u, v in arc_pairs:
            if u in node_map and v in node_map:
                filtered_arc_pairs.append([node_map[u], node_map[v]])

        # Mock metrics
        metrics = {
            "top1_accuracy": 0.884,
            "route_continuity": 0.942,
            "n_gps_points": len(gps_points),
            "n_predicted_arcs": len(predicted_arcs),
            "inference_time_sec": 1.25,
        }

        return {
            "trip_id": trip_id,
            "gps_points": gps_points,
            "predicted_arcs": predicted_arcs,
            "predicted_arc_coords": predicted_arc_coords,
            "road_network": {
                "node_coords": filtered_node_coords,
                "arc_pairs": filtered_arc_pairs,
            },
            "metrics": metrics,
        }

    def _generate_synthetic_trip(self, trip_id: str) -> Dict[str, Any]:
        """Fallback generator if local data files are not present."""
        base_lon, base_lat = 2.418, 47.099
        gps_points = []
        node_coords = []
        arc_pairs = []
        predicted_arc_coords = []
        predicted_arcs = []

        n_pts = 35
        for i in range(n_pts):
            lon = base_lon + (i * 0.0004) + ((i % 3) * 0.00005)
            lat = base_lat + (i * 0.0003) - ((i % 2) * 0.00004)
            t = float(i * 3)
            gps_points.append([round(lon, 6), round(lat, 6), t])
            node_coords.append([round(lon, 6), round(lat, 6)])

        for i in range(n_pts - 1):
            arc_pairs.append([i, i + 1])
            predicted_arcs.append(100 + i)
            predicted_arc_coords.append([node_coords[i], node_coords[i + 1]])

        return {
            "trip_id": trip_id,
            "gps_points": gps_points,
            "predicted_arcs": predicted_arcs,
            "predicted_arc_coords": predicted_arc_coords,
            "road_network": {
                "node_coords": node_coords,
                "arc_pairs": arc_pairs,
            },
            "metrics": {
                "top1_accuracy": 0.875,
                "route_continuity": 0.920,
                "n_gps_points": len(gps_points),
                "n_predicted_arcs": len(predicted_arcs),
                "inference_time_sec": 0.45,
            },
        }
