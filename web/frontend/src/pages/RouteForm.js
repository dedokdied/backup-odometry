import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { routesApi } from "../api/client";
export default function RouteForm({ mode }) {
    const { id } = useParams();
    const nav = useNavigate();
    const [name, setName] = useState("");
    const [startPoint, setStartPoint] = useState("");
    const [endPoint, setEndPoint] = useState("");
    const [err, setErr] = useState("");
    useEffect(() => {
        if (mode === "edit" && id)
            routesApi.get(Number(id)).then((r) => {
                setName(r.name);
                setStartPoint(r.startPoint);
                setEndPoint(r.endPoint);
            }).catch((e) => setErr(e.message));
    }, [mode, id]);
    const submit = async (e) => {
        e.preventDefault();
        try {
            if (mode === "new") {
                const r = await routesApi.create({ name, startPoint, endPoint });
                nav(`/routes/${r.id}`);
            }
            else {
                await routesApi.update(Number(id), { name, startPoint, endPoint });
                nav(`/routes/${id}`);
            }
        }
        catch (e2) {
            setErr(e2.message);
        }
    };
    return (_jsxs("div", { children: [_jsx("h2", { children: mode === "new" ? "Новый маршрут" : `Маршрут ${id}` }), err && _jsx("p", { className: "err", children: err }), _jsxs("form", { onSubmit: submit, children: [_jsxs("label", { children: ["\u041D\u0430\u0437\u0432\u0430\u043D\u0438\u0435", _jsx("input", { value: name, onChange: (e) => setName(e.target.value), required: true })] }), _jsxs("label", { children: ["\u041E\u0442\u043A\u0443\u0434\u0430", _jsx("input", { value: startPoint, onChange: (e) => setStartPoint(e.target.value) })] }), _jsxs("label", { children: ["\u041A\u0443\u0434\u0430", _jsx("input", { value: endPoint, onChange: (e) => setEndPoint(e.target.value) })] }), _jsx("button", { type: "submit", children: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C" }), " ", _jsx(Link, { to: "/routes", children: "\u041E\u0442\u043C\u0435\u043D\u0430" })] })] }));
}
