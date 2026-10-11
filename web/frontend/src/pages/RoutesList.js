import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { routesApi } from "../api/client";
export default function RoutesList() {
    const [routes, setRoutes] = useState([]);
    const [err, setErr] = useState("");
    const [loading, setLoading] = useState(true);
    const nav = useNavigate();
    useEffect(() => {
        routesApi.list().then(setRoutes).catch((e) => setErr(e.message)).finally(() => setLoading(false));
    }, []);
    const onDelete = async (id) => {
        if (!window.confirm(`Удалить маршрут ${id}?`))
            return;
        try {
            await routesApi.remove(id);
            setRoutes(routes.filter((r) => r.id !== id));
        }
        catch (e) {
            setErr(e.message);
        }
    };
    if (loading)
        return _jsx("p", { children: "\u0417\u0430\u0433\u0440\u0443\u0437\u043A\u0430\u2026" });
    return (_jsxs("div", { children: [_jsx("h2", { children: "\u041C\u0430\u0440\u0448\u0440\u0443\u0442\u044B" }), err && _jsx("p", { className: "err", children: err }), _jsx("button", { onClick: () => nav("/routes/new"), children: "\u0414\u043E\u0431\u0430\u0432\u0438\u0442\u044C \u043C\u0430\u0440\u0448\u0440\u0443\u0442" }), _jsxs("table", { children: [_jsx("thead", { children: _jsxs("tr", { children: [_jsx("th", { children: "ID" }), _jsx("th", { children: "\u041D\u0430\u0437\u0432\u0430\u043D\u0438\u0435" }), _jsx("th", { children: "\u041E\u0442\u043A\u0443\u0434\u0430" }), _jsx("th", { children: "\u041A\u0443\u0434\u0430" }), _jsx("th", {})] }) }), _jsx("tbody", { children: routes.map((r) => (_jsxs("tr", { children: [_jsx("td", { children: r.id }), _jsx("td", { children: _jsx(Link, { to: `/routes/${r.id}`, children: r.name }) }), _jsx("td", { children: r.startPoint }), _jsx("td", { children: r.endPoint }), _jsxs("td", { children: [_jsx(Link, { to: `/routes/${r.id}/edit`, children: "\u0418\u0437\u043C\u0435\u043D\u0438\u0442\u044C" }), " ", _jsx("button", { onClick: () => onDelete(r.id), children: "\u0423\u0434\u0430\u043B\u0438\u0442\u044C" })] })] }, r.id))) })] })] }));
}
