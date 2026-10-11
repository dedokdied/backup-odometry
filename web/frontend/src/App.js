import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom";
import Nav from "./components/Nav";
import RouteForm from "./pages/RouteForm";
import RouteView from "./pages/RouteView";
import RoutesList from "./pages/RoutesList";
import RunForm from "./pages/RunForm";
import RunView from "./pages/RunView";
import RunsList from "./pages/RunsList";
import TripPage from "./pages/TripPage";
import ReportPage from "./pages/ReportPage";
import LoginPage from "./pages/LoginPage";
import SimulatorPage from "./pages/SimulatorPage";
export default function App() {
    return (_jsxs(BrowserRouter, { children: [_jsx(Nav, {}), _jsx("main", { children: _jsxs(Routes, { children: [_jsx(Route, { path: "/", element: _jsx(Navigate, { to: "/runs" }) }), _jsx(Route, { path: "/login", element: _jsx(LoginPage, {}) }), _jsx(Route, { path: "/runs", element: _jsx(RunsList, {}) }), _jsx(Route, { path: "/runs/new", element: _jsx(RunForm, { mode: "new" }) }), _jsx(Route, { path: "/runs/:id", element: _jsx(RunView, {}) }), _jsx(Route, { path: "/runs/:id/trip", element: _jsx(TripPage, {}) }), _jsx(Route, { path: "/runs/:id/report", element: _jsx(ReportPage, {}) }), _jsx(Route, { path: "/runs/:id/edit", element: _jsx(RunForm, { mode: "edit" }) }), _jsx(Route, { path: "/routes", element: _jsx(RoutesList, {}) }), _jsx(Route, { path: "/routes/new", element: _jsx(RouteForm, { mode: "new" }) }), _jsx(Route, { path: "/routes/:id", element: _jsx(RouteView, {}) }), _jsx(Route, { path: "/routes/:id/edit", element: _jsx(RouteForm, { mode: "edit" }) }), _jsx(Route, { path: "/simulator", element: _jsx(SimulatorPage, {}) })] }) })] }));
}
