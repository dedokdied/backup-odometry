import { jsx as _jsx, jsxs as _jsxs } from "react/jsx-runtime";
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { authApi } from "../api/client";
import "./LoginPage.css";
export default function LoginPage() {
    const nav = useNavigate();
    const [username, setUsername] = useState("");
    const [password, setPassword] = useState("");
    const [err, setErr] = useState("");
    const [mode, setMode] = useState("login");
    const submit = async (e) => {
        e.preventDefault();
        setErr("");
        try {
            const res = mode === "login"
                ? await authApi.login({ username, password })
                : await authApi.register({ username, password });
            localStorage.setItem("token", res.token);
            nav("/runs");
        }
        catch (e) {
            setErr(e.message || "Ошибка");
        }
    };
    return (_jsx("div", { className: "login-wrap", children: _jsxs("div", { className: "login-card", children: [_jsx("h2", { children: mode === "login" ? "Вход" : "Регистрация" }), err && _jsx("p", { className: "err", children: err }), _jsxs("form", { onSubmit: submit, children: [_jsxs("label", { children: ["\u041B\u043E\u0433\u0438\u043D", _jsx("input", { value: username, onChange: (e) => setUsername(e.target.value), required: true })] }), _jsxs("label", { children: ["\u041F\u0430\u0440\u043E\u043B\u044C", _jsx("input", { type: "password", value: password, onChange: (e) => setPassword(e.target.value), required: true })] }), _jsx("button", { type: "submit", children: mode === "login" ? "Войти" : "Зарегистрироваться" })] }), _jsxs("p", { children: [mode === "login" ? "Нет аккаунта?" : "Уже есть аккаунт?", " ", _jsx("button", { type: "button", onClick: () => setMode(mode === "login" ? "register" : "login"), children: mode === "login" ? "Регистрация" : "Войти" })] }), _jsxs("p", { className: "demo", children: ["\u0414\u0435\u043C\u043E: ", _jsx("code", { children: "admin / admin123" }), " \u0438\u043B\u0438 ", _jsx("code", { children: "user / user123" })] })] }) }));
}
