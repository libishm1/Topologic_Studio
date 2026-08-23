import React from "react";
import ReactDOM from "react-dom/client";

import App from "./App.jsx";
import "./styles/tokens.css";
import "./styles/base.css";
import "./styles/app.css";

// No StrictMode. It double-invokes effects in development, which for a WebGL
// context plus a fragments worker means the engine is created, torn down and
// recreated on every mount - the single biggest source of viewer flakiness in
// the Classic build. React 19's other dev warnings still apply.
ReactDOM.createRoot(document.getElementById("root")).render(<App />);
