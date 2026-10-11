import { jsx as _jsx, Fragment as _Fragment, jsxs as _jsxs } from "react/jsx-runtime";
import { useRef } from "react";
import { MapContainer, TileLayer, Polyline, CircleMarker, useMapEvents } from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
    iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
    iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
    shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});
function FitBounds({ frames, mapRef }) {
    useMapEvents({
        moveend() {
            if (frames.length) {
                const bounds = L.latLngBounds(frames.map((f) => [f.y, f.x]));
                if (bounds.isValid())
                    mapRef.current?.fitBounds(bounds, { padding: [20, 20] });
            }
        },
    });
    return null;
}
function ModelPath({ frames }) {
    const points = frames.map((f) => [f.y, f.x]);
    return _jsx(Polyline, { positions: points, color: "#0066cc", weight: 3, opacity: 0.8 });
}
function GNSSPath({ frames }) {
    const points = frames
        .filter((f) => f.v_gnss !== null && f.v_gnss !== undefined)
        .map((f) => [f.y, f.x]);
    return _jsx(Polyline, { positions: points, color: "#cc0000", weight: 2, opacity: 0.7, dashArray: "5,5" });
}
function CurrentMarker({ frame, showGnss }) {
    if (!frame)
        return null;
    return (_jsxs(_Fragment, { children: [_jsx(CircleMarker, { center: [frame.y, frame.x], radius: 8, color: "#0066cc", fillColor: "#0066cc", fillOpacity: 0.9, stroke: true, weight: 2 }), showGnss && frame.v_gnss !== null && frame.v_gnss !== undefined && (_jsx(CircleMarker, { center: [frame.y, frame.x], radius: 6, color: "#cc0000", fillColor: "#cc0000", fillOpacity: 0.7, stroke: true, weight: 1 }))] }));
}
export function TripMap({ frames, currentIdx, showGnss = true, showModel = true, }) {
    const mapRef = useRef(null);
    if (!frames.length)
        return _jsx("div", { style: { height: 300, background: "#f0f0f0" } });
    const center = [frames[0].y, frames[0].x];
    const currentFrame = frames[currentIdx] ?? frames[frames.length - 1];
    return (_jsxs(MapContainer, { ref: mapRef, center: center, zoom: 16, style: { height: "100%", width: "100%" }, scrollWheelZoom: true, children: [_jsx(TileLayer, { attribution: '\u00A9 <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors', url: "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png" }), showModel && _jsx(ModelPath, { frames: frames }), showGnss && _jsx(GNSSPath, { frames: frames }), _jsx(CurrentMarker, { frame: currentFrame, showGnss: showGnss }), _jsx(FitBounds, { frames: frames, mapRef: mapRef })] }));
}
