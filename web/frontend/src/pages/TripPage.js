import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState, useCallback } from "react";
import { useParams, Link } from "react-router-dom";
import { SpeedChart, TrustChart, PositionChart } from "../components/Charts";
import { TripMap } from "../components/MapView";
import { MetricsBar } from "../components/MetricsBar";
import { Pedal } from "../components/Pedal";
import { runsApi } from "../api/client";
import "./TripPage.css";
export default function TripPage() {
    const { id } = useParams();
    const [frames, setFrames] = useState([]);
    const [currentIdx, setCurrentIdx] = useState(0);
    const [playing, setPlaying] = useState(false);
    const [speed, setSpeed] = useState(1);
    const [showGnss, setShowGnss] = useState(true);
    const [showModel, setShowModel] = useState(true);
    const [intervention, setIntervention] = useState(false);
    const [interventionActive, setInterventionActive] = useState(false);
    const [pedalValue, setPedalValue] = useState(0);
    const [loading, setLoading] = useState(true);
    const [err, setErr] = useState("");
    const load = useCallback(async () => {
        if (!id)
            return;
        setLoading(true);
        try {
            const telemetry = await runsApi.telemetry(id, 0, 1e9, 1);
            setFrames(telemetry);
            setLoading(false);
        }
        catch (e) {
            setErr(e.message);
            setLoading(false);
        }
    }, [id]);
    useEffect(() => { load(); }, [load]);
    useEffect(() => {
        if (!playing || currentIdx >= frames.length - 1)
            return;
        const dt = (frames[currentIdx + 1]?.t ?? 0.02) - (frames[currentIdx]?.t ?? 0);
        const timer = setTimeout(() => setCurrentIdx((i) => Math.min(i + 1, frames.length - 1)), dt * 1000 / speed);
        return () => clearTimeout(timer);
    }, [playing, currentIdx, frames, speed]);
    const handlePedal = (v) => {
        setPedalValue(v);
        if (intervention) {
            const f = frames[currentIdx];
            if (f) {
                const arr = [...frames];
                arr[currentIdx] = { ...f, u: v };
                setFrames(arr);
            }
        }
    };
    if (loading)
        return _jsx("div", { className: "trip", children: "\u0417\u0430\u0433\u0440\u0443\u0437\u043A\u0430\u2026" });
    if (err)
        return _jsx("div", { className: "trip err", children: err });
    return (_jsxs("div", { className: "trip", children: [_jsxs("header", { className: "trip-header", children: [_jsx(Link, { to: "/runs", className: "back", children: "\u2190 \u041A \u043F\u0440\u043E\u0433\u043E\u043D\u0430\u043C" }), _jsxs("h2", { children: ["\u041F\u0440\u043E\u0433\u043E\u043D ", id] }), _jsxs("div", { className: "controls", children: [_jsx("button", { onClick: () => setPlaying(!playing), children: playing ? "Пауза" : "Старт" }), _jsxs("select", { value: speed, onChange: (e) => setSpeed(Number(e.target.value)), children: [_jsx("option", { value: 0.5, children: "0.5\u00D7" }), _jsx("option", { value: 1, children: "1\u00D7" }), _jsx("option", { value: 2, children: "2\u00D7" }), _jsx("option", { value: 4, children: "4\u00D7" })] }), _jsxs("label", { children: [_jsx("input", { type: "checkbox", checked: showGnss, onChange: (e) => setShowGnss(e.target.checked) }), " GNSS"] }), _jsxs("label", { children: [_jsx("input", { type: "checkbox", checked: showModel, onChange: (e) => setShowModel(e.target.checked) }), " \u041C\u043E\u0434\u0435\u043B\u044C"] }), _jsx("button", { onClick: () => setCurrentIdx(0), children: "\u0421\u0431\u0440\u043E\u0441" }), _jsx("button", { className: intervention ? "active" : "", onClick: () => setIntervention(!intervention), children: intervention ? "Рулю" : "Смотрю" }), _jsx(Link, { to: `/runs/${id}/report`, style: { marginLeft: 12, textDecoration: "none", color: "#0066cc" }, children: "\u041E\u0442\u0447\u0451\u0442" })] })] }), _jsxs("main", { className: "trip-main", children: [_jsxs("aside", { className: "charts", children: [_jsx(SpeedChart, { frames: frames }), _jsx(TrustChart, { frames: frames }), _jsx(PositionChart, { frames: frames })] }), _jsx("section", { className: "map-area", children: _jsx(TripMap, { frames: frames, currentIdx: currentIdx, showGnss: showGnss, showModel: showModel }) })] }), _jsx(MetricsBar, { frames: frames, currentIdx: currentIdx, intervention: interventionActive }), intervention && (_jsxs("div", { className: "pedal-overlay", children: [_jsx(Pedal, { value: pedalValue, onChange: handlePedal, intervention: true, interventionActive: interventionActive }), _jsx("button", { className: interventionActive ? "active" : "", onClick: () => setInterventionActive(!interventionActive), children: interventionActive ? "Зафиксировать педаль" : "Активировать педаль" })] }))] }));
}
