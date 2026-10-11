import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { runsApi } from "../api/client";
export default function RunView() {
    const { id } = useParams();
    const nav = useNavigate();
    const [run, setRun] = useState(null);
    const [err, setErr] = useState("");
    useEffect(() => {
        runsApi.get(Number(id)).then(setRun).catch((e) => setErr(e.message));
    }, [id]);
    const onDelete = async () => {
        if (!window.confirm(`Удалить прогон ${id}?`))
            return;
        try {
            await runsApi.remove(Number(id));
            nav("/runs");
        }
        catch (e) {
            setErr(e.message);
        }
    };
    if (err)
        return _jsx("p", { className: "err", children: err });
    if (!run)
        return _jsx("p", { children: "\u0417\u0430\u0433\u0440\u0443\u0437\u043A\u0430\u2026" });
    return (_jsxs("div", { children: [_jsx("h2", { children: run.title }), _jsxs("dl", { children: [_jsx("dt", { children: "ID" }), _jsx("dd", { children: run.id }), _jsx("dt", { children: "\u041C\u0430\u0440\u0448\u0440\u0443\u0442" }), _jsx("dd", { children: run.routeId }), _jsx("dt", { children: "\u041E\u043F\u0438\u0441\u0430\u043D\u0438\u0435" }), _jsx("dd", { children: run.description }), _jsx("dt", { children: "\u0414\u0430\u0442\u0430" }), _jsx("dd", { children: run.date }), _jsx("dt", { children: "\u0414\u043B\u0438\u0442\u0435\u043B\u044C\u043D\u043E\u0441\u0442\u044C, \u0441" }), _jsx("dd", { children: run.duration }), _jsx("dt", { children: "\u0414\u043B\u0438\u043D\u0430, \u043C" }), _jsx("dd", { children: run.length }), _jsx("dt", { children: "\u0421\u0442\u0430\u0442\u0443\u0441" }), _jsx("dd", { children: run.status }), _jsx("dt", { children: "\u0424\u0430\u0439\u043B" }), _jsx("dd", { children: run.sourceFile })] }), _jsx(Link, { to: `/runs/${run.id}/edit`, children: "\u0418\u0437\u043C\u0435\u043D\u0438\u0442\u044C" }), " ", _jsx(Link, { to: `/runs/${run.id}/trip`, children: "\u041F\u043E\u0435\u0437\u0434\u043A\u0430" }), " ", _jsx("button", { onClick: onDelete, children: "\u0423\u0434\u0430\u043B\u0438\u0442\u044C" }), " ", _jsx(Link, { to: "/runs", children: "\u041A \u0441\u043F\u0438\u0441\u043A\u0443" })] }));
}
