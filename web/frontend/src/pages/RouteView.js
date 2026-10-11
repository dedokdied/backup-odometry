import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { routesApi } from "../api/client";
export default function RouteView() {
    const { id } = useParams();
    const nav = useNavigate();
    const [route, setRoute] = useState(null);
    const [err, setErr] = useState("");
    useEffect(() => {
        routesApi.get(Number(id)).then(setRoute).catch((e) => setErr(e.message));
    }, [id]);
    const onDelete = async () => {
        if (!window.confirm(`Удалить маршрут ${id}?`))
            return;
        try {
            await routesApi.remove(Number(id));
            nav("/routes");
        }
        catch (e) {
            setErr(e.message);
        }
    };
    if (err)
        return _jsx("p", { className: "err", children: err });
    if (!route)
        return _jsx("p", { children: "\u0417\u0430\u0433\u0440\u0443\u0437\u043A\u0430\u2026" });
    return (_jsxs("div", { children: [_jsx("h2", { children: route.name }), _jsxs("dl", { children: [_jsx("dt", { children: "ID" }), _jsx("dd", { children: route.id }), _jsx("dt", { children: "\u041E\u0442\u043A\u0443\u0434\u0430" }), _jsx("dd", { children: route.startPoint }), _jsx("dt", { children: "\u041A\u0443\u0434\u0430" }), _jsx("dd", { children: route.endPoint }), _jsx("dt", { children: "\u041E\u0441\u0442\u0430\u043D\u043E\u0432\u043A\u0438" }), _jsx("dd", { children: route.stops.join(", ") })] }), _jsx(Link, { to: `/routes/${route.id}/edit`, children: "\u0418\u0437\u043C\u0435\u043D\u0438\u0442\u044C" }), " ", _jsx("button", { onClick: onDelete, children: "\u0423\u0434\u0430\u043B\u0438\u0442\u044C" }), " ", _jsx(Link, { to: "/routes", children: "\u041A \u0441\u043F\u0438\u0441\u043A\u0443" })] }));
}
