import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { runsApi } from "../api/client";
export default function RunForm({ mode }) {
    const { id } = useParams();
    const nav = useNavigate();
    const [title, setTitle] = useState("");
    const [routeId, setRouteId] = useState(1);
    const [description, setDescription] = useState("");
    const [err, setErr] = useState("");
    useEffect(() => {
        if (mode === "edit" && id)
            runsApi.get(Number(id)).then((r) => {
                setTitle(r.title);
                setRouteId(r.routeId);
                setDescription(r.description);
            }).catch((e) => setErr(e.message));
    }, [mode, id]);
    const submit = async (e) => {
        e.preventDefault();
        try {
            if (mode === "new") {
                const r = await runsApi.create({ title, routeId, description });
                nav(`/runs/${r.id}`);
            }
            else {
                await runsApi.update(Number(id), { title, routeId, description });
                nav(`/runs/${id}`);
            }
        }
        catch (e2) {
            setErr(e2.message);
        }
    };
    return (_jsxs("div", { children: [_jsx("h2", { children: mode === "new" ? "Новый прогон" : `Прогон ${id}` }), err && _jsx("p", { className: "err", children: err }), _jsxs("form", { onSubmit: submit, children: [_jsxs("label", { children: ["\u041D\u0430\u0437\u0432\u0430\u043D\u0438\u0435", _jsx("input", { value: title, onChange: (e) => setTitle(e.target.value), required: true })] }), _jsxs("label", { children: ["ID \u043C\u0430\u0440\u0448\u0440\u0443\u0442\u0430", _jsx("input", { type: "number", value: routeId, onChange: (e) => setRouteId(Number(e.target.value)), required: true })] }), _jsxs("label", { children: ["\u041E\u043F\u0438\u0441\u0430\u043D\u0438\u0435", _jsx("input", { value: description, onChange: (e) => setDescription(e.target.value) })] }), _jsx("button", { type: "submit", children: "\u0421\u043E\u0445\u0440\u0430\u043D\u0438\u0442\u044C" }), " ", _jsx(Link, { to: "/runs", children: "\u041E\u0442\u043C\u0435\u043D\u0430" })] })] }));
}
