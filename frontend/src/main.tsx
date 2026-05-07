import { createRoot } from "react-dom/client";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { RouterProvider } from "@tanstack/react-router";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { router } from "./router";
import { GOOGLE_CLIENT_ID } from "./lib/constants";
import "./index.css";

// React StrictMode is intentionally disabled.
// In development it mounts → unmounts → mounts every component twice to
// surface side-effects.  This causes Google's GSI SDK to call initialize()
// twice, which logs a warning and makes the sign-in button unreliable.

const queryClient = new QueryClient({
  defaultOptions: {
    queries: { staleTime: 60_000, retry: 1 },
  },
});

createRoot(document.getElementById("root")!).render(
  <GoogleOAuthProvider clientId={GOOGLE_CLIENT_ID}>
    <QueryClientProvider client={queryClient}>
      <RouterProvider router={router} />
    </QueryClientProvider>
  </GoogleOAuthProvider>
);
