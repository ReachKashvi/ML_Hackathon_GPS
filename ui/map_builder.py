"""
PathSnap UI - Folium Map Builder
Constructs the interactive Folium map visualization from the API response payload.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional
import folium


def build_prediction_map(
    api_response: Dict[str, Any],
    colors: Optional[Dict[str, str]] = None,
    zoom_start: int = 14,
    tiles: str = "OpenStreetMap",
) -> folium.Map:
    """
    Builds a Folium Map visualization displaying:
    1. Surrounding Road Network (gray polylines)
    2. GPS Observations (blue circle markers)
    3. Predicted Route (thick red polylines)
    4. Start (green) and End (red) markers

    All layers are toggleable using Folium LayerControl.
    """
    if colors is None:
        colors = {
            "gps_points": "#2E7DFF",
            "predicted_route": "#E24B4A",
            "road_network": "#999999",
        }

    gps_points: List[List[float]] = api_response.get("gps_points", [])
    predicted_arc_coords: List[List[List[float]]] = api_response.get("predicted_arc_coords", [])
    predicted_arcs: List[int] = api_response.get("predicted_arcs", [])
    road_net: Dict[str, Any] = api_response.get("road_network", {})
    node_coords: List[List[float]] = road_net.get("node_coords", [])
    arc_pairs: List[List[int]] = road_net.get("arc_pairs", [])

    # Calculate center and bounds from GPS points (or fallback)
    if gps_points:
        lons = [p[0] for p in gps_points]
        lats = [p[1] for p in gps_points]
        min_lat, max_lat = min(lats), max(lats)
        min_lon, max_lon = min(lons), max(lons)
        center_lat = (min_lat + max_lat) / 2.0
        center_lon = (min_lon + max_lon) / 2.0
    else:
        min_lat, max_lat = 47.08, 47.12
        min_lon, max_lon = 2.38, 2.45
        center_lat, center_lon = 47.10, 2.41

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=zoom_start,
        tiles=tiles,
        control_scale=True,
    )

    # 1. Road Network Layer (thin gray lines)
    fg_road = folium.FeatureGroup(name="Road network", show=True)
    if node_coords and arc_pairs:
        road_segments: List[List[List[float]]] = []
        for u, v in arc_pairs:
            if 0 <= u < len(node_coords) and 0 <= v < len(node_coords):
                p1 = [node_coords[u][1], node_coords[u][0]]  # [lat, lon]
                p2 = [node_coords[v][1], node_coords[v][0]]  # [lat, lon]
                road_segments.append([p1, p2])

        if road_segments:
            folium.PolyLine(
                locations=road_segments,
                color=colors.get("road_network", "#999999"),
                weight=1.5,
                opacity=0.6,
                tooltip="Road Network Segment",
            ).add_to(fg_road)
    fg_road.add_to(m)

    # 2. Predicted Route Layer (thick red lines)
    fg_route = folium.FeatureGroup(name="Predicted route", show=True)
    for idx, arc_pts in enumerate(predicted_arc_coords):
        if len(arc_pts) >= 2:
            leaflet_pts = [[pt[1], pt[0]] for pt in arc_pts]  # convert [lon, lat] -> [lat, lon]
            arc_id_str = f"Arc #{predicted_arcs[idx]}" if idx < len(predicted_arcs) else f"Step {idx + 1}"
            folium.PolyLine(
                locations=leaflet_pts,
                color=colors.get("predicted_route", "#E24B4A"),
                weight=5,
                opacity=0.85,
                tooltip=f"Predicted Route: {arc_id_str}",
            ).add_to(fg_route)
    fg_route.add_to(m)

    # 3. GPS Observations Layer (blue CircleMarkers)
    fg_gps = folium.FeatureGroup(name="GPS observations", show=True)
    for idx, pt in enumerate(gps_points):
        lat, lon = pt[1], pt[0]
        t_sec = pt[2] if len(pt) > 2 else idx
        folium.CircleMarker(
            location=[lat, lon],
            radius=3,
            color=colors.get("gps_points", "#2E7DFF"),
            fill=True,
            fill_color=colors.get("gps_points", "#2E7DFF"),
            fill_opacity=0.85,
            tooltip=f"GPS #{idx + 1}: ({lat:.6f}, {lon:.6f}) @ {t_sec}s",
        ).add_to(fg_gps)
    fg_gps.add_to(m)

    # 4. Start & End Markers (green and red)
    fg_endpoints = folium.FeatureGroup(name="Start / End points", show=True)
    if gps_points:
        start_pt = gps_points[0]
        end_pt = gps_points[-1]

        folium.Marker(
            location=[start_pt[1], start_pt[0]],
            popup=f"<b>Trip Start</b><br>Trip: {api_response.get('trip_id', 'N/A')}<br>Coords: {start_pt[1]:.5f}, {start_pt[0]:.5f}",
            tooltip="Trip Start (Origin)",
            icon=folium.Icon(color="green", icon="play", prefix="fa"),
        ).add_to(fg_endpoints)

        folium.Marker(
            location=[end_pt[1], end_pt[0]],
            popup=f"<b>Trip End</b><br>Coords: {end_pt[1]:.5f}, {end_pt[0]:.5f}",
            tooltip="Trip End (Destination)",
            icon=folium.Icon(color="red", icon="stop", prefix="fa"),
        ).add_to(fg_endpoints)
    fg_endpoints.add_to(m)

    # Fit bounds to GPS observations
    if gps_points:
        m.fit_bounds([[min_lat, min_lon], [max_lat, max_lon]])

    # Toggle control for all layers
    folium.LayerControl(collapsed=False).add_to(m)

    return m
