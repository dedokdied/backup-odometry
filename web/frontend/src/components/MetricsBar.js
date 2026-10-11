import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
export function MetricsBar({ frames, currentIdx, intervention, }) {
    if (!frames.length)
        return _jsx("div", { className: "metrics", children: "\u041D\u0435\u0442 \u0434\u0430\u043D\u043D\u044B\u0445" });
    const f = frames[currentIdx] ?? frames[frames.length - 1];
    const vErrM = frames.reduce((a, b) => a + Math.pow(b.v_model - (b.v_gnss ?? b.v_model), 2), 0);
    const vErrW = frames.reduce((a, b) => a + Math.pow(b.v_wheels - (b.v_gnss ?? b.v_wheels), 2), 0);
    const n = frames.length;
    const rmseM = Math.sqrt(vErrM / n);
    const rmseW = Math.sqrt(vErrW / n);
    const win = rmseW - rmseM;
    return (_jsxs("div", { className: "metrics", children: [_jsxs("div", { className: "row", children: [_jsx("span", { children: "\u0422\u0435\u043A\u0443\u0449\u0430\u044F \u0441\u043A\u043E\u0440\u043E\u0441\u0442\u044C:" }), _jsxs("strong", { children: [f.v_model.toFixed(2), " \u043C/\u0441"] })] }), _jsxs("div", { className: "row", children: [_jsx("span", { children: "RMSE \u043C\u043E\u0434\u0435\u043B\u044C:" }), _jsxs("strong", { children: [rmseM.toFixed(2), " \u043C/\u0441"] })] }), _jsxs("div", { className: "row", children: [_jsx("span", { children: "RMSE \u043A\u043E\u043B\u0451\u0441\u0430:" }), _jsxs("strong", { children: [rmseW.toFixed(2), " \u043C/\u0441"] })] }), _jsxs("div", { className: "row", children: [_jsx("span", { children: "\u0412\u044B\u0438\u0433\u0440\u044B\u0448:" }), _jsxs("strong", { style: { color: win >= 0 ? "#006600" : "#cc0000" }, children: [win >= 0 ? "+" : "", win.toFixed(2), " \u043C/\u0441"] })] }), _jsxs("div", { className: "row", children: [_jsx("span", { children: "\u0414\u043E\u0432\u0435\u0440\u0438\u0435:" }), _jsxs("strong", { children: [(f.trust * 100).toFixed(0), "%"] })] }), _jsxs("div", { className: "row", children: [_jsx("span", { children: "\u0421\u043A\u043E\u043B\u044C\u0436\u0435\u043D\u0438\u0435:" }), _jsxs("strong", { children: [(f.slip * 100).toFixed(0), "%"] })] }), _jsxs("div", { className: "row", children: [_jsx("span", { children: "\u041F\u0440\u043E\u0439\u0434\u0435\u043D\u043D\u044B\u0439 \u043F\u0443\u0442\u044C:" }), _jsxs("strong", { children: [f.s.toFixed(0), " \u043C"] })] }), intervention && (_jsx("div", { className: "intervention", children: "\u26A0 \u0420\u0435\u0436\u0438\u043C \u00AB\u0420\u0443\u043B\u044E\u00BB \u2014 \u0431\u0435\u043D\u0447\u043C\u0430\u0440\u043A \u0433\u043B\u044E\u0447\u0438\u0442 (\u044D\u0442\u0430\u043B\u043E\u043D \u2014 \u0437\u0430\u043F\u0438\u0441\u044C)" }))] }));
}
