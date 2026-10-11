import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, } from "recharts";
export function SpeedChart({ frames }) {
    const data = frames.map((f) => ({
        t: f.t,
        "Модель": +f.v_model.toFixed(2),
        "GNSS": f.v_gnss ? +f.v_gnss.toFixed(2) : null,
        "Колёса": +f.v_wheels.toFixed(2),
        "Педаль": f.u,
    }));
    return (_jsx(ResponsiveContainer, { width: "100%", height: 200, children: _jsxs(LineChart, { data: data, margin: { top: 5, right: 30, left: 20, bottom: 5 }, children: [_jsx(CartesianGrid, { strokeDasharray: "3 3" }), _jsx(XAxis, { dataKey: "t", tickFormatter: (v) => `${v.toFixed(0)}с` }), _jsx(YAxis, {}), _jsx(Tooltip, {}), _jsx(Legend, {}), _jsx(Line, { type: "monotone", dataKey: "\u041C\u043E\u0434\u0435\u043B\u044C", stroke: "#0066cc", strokeWidth: 2, dot: false }), _jsx(Line, { type: "monotone", dataKey: "GNSS", stroke: "#cc0000", strokeWidth: 2, dot: false }), _jsx(Line, { type: "monotone", dataKey: "\u041A\u043E\u043B\u0451\u0441\u0430", stroke: "#888", strokeWidth: 1.5, strokeDasharray: "4 4", dot: false }), _jsx(Line, { type: "monotone", dataKey: "\u041F\u0435\u0434\u0430\u043B\u044C", stroke: "#009933", strokeWidth: 1, dot: false })] }) }));
}
export function TrustChart({ frames }) {
    const data = frames.map((f) => ({
        t: f.t,
        "Доверие": +f.trust.toFixed(2),
        "Скольжение": +f.slip.toFixed(2),
    }));
    return (_jsx(ResponsiveContainer, { width: "100%", height: 120, children: _jsxs(LineChart, { data: data, margin: { top: 5, right: 30, left: 20, bottom: 5 }, children: [_jsx(CartesianGrid, { strokeDasharray: "3 3" }), _jsx(XAxis, { dataKey: "t", tickFormatter: (v) => `${v.toFixed(0)}с` }), _jsx(YAxis, { domain: [0, 1] }), _jsx(Tooltip, {}), _jsx(Legend, {}), _jsx(Line, { type: "monotone", dataKey: "\u0414\u043E\u0432\u0435\u0440\u0438\u0435", stroke: "#0066cc", strokeWidth: 2, dot: false }), _jsx(Line, { type: "monotone", dataKey: "\u0421\u043A\u043E\u043B\u044C\u0436\u0435\u043D\u0438\u0435", stroke: "#cc0000", strokeWidth: 2, dot: false })] }) }));
}
export function PositionChart({ frames }) {
    const data = frames.map((f) => ({
        t: f.t,
        X: +f.x.toFixed(1),
        Y: +f.y.toFixed(1),
        S: +f.s.toFixed(1),
    }));
    return (_jsx(ResponsiveContainer, { width: "100%", height: 120, children: _jsxs(LineChart, { data: data, margin: { top: 5, right: 30, left: 20, bottom: 5 }, children: [_jsx(CartesianGrid, { strokeDasharray: "3 3" }), _jsx(XAxis, { dataKey: "t", tickFormatter: (v) => `${v.toFixed(0)}с` }), _jsx(YAxis, {}), _jsx(Tooltip, {}), _jsx(Legend, {}), _jsx(Line, { type: "monotone", dataKey: "S", stroke: "#0066cc", strokeWidth: 2, dot: false }), _jsx(Line, { type: "monotone", dataKey: "X", stroke: "#cc0000", strokeWidth: 1.5, dot: false }), _jsx(Line, { type: "monotone", dataKey: "Y", stroke: "#009933", strokeWidth: 1.5, dot: false })] }) }));
}
