import { StrictMode } from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import { initSession } from "./api/client";

async function bootstrap() {
  // Acquire the session token before rendering so the first state-changing
  // request carries it. Render regardless if it fails (read-only UI still works).
  try {
    await initSession();
  } catch (err) {
    console.error("Failed to initialize session token", err);
  }
  const root = document.getElementById("root");
  if (root) {
    createRoot(root).render(
      <StrictMode>
        <App />
      </StrictMode>
    );
  }
}

bootstrap();
