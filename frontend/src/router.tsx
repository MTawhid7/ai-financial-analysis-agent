import { createRouter, createRoute, createRootRoute, redirect } from "@tanstack/react-router";
import { LoginPage } from "./pages/LoginPage";
import { ChatPage } from "./pages/ChatPage";
import { getMe } from "./lib/api";

const rootRoute = createRootRoute();

const loginRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/",
  component: LoginPage,
});

const chatRoute = createRoute({
  getParentRoute: () => rootRoute,
  path: "/chat",
  beforeLoad: async () => {
    try {
      await getMe();
    } catch {
      throw redirect({ to: "/" });
    }
  },
  component: ChatPage,
});

const routeTree = rootRoute.addChildren([loginRoute, chatRoute]);

export const router = createRouter({ routeTree });

declare module "@tanstack/react-router" {
  interface Register {
    router: typeof router;
  }
}
