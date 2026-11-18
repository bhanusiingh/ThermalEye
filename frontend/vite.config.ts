import { resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const rootDir = fileURLToPath(new URL(".", import.meta.url));

function dashboardRoutePlugin() {
  const rewriteDashboard = (req: any, _res: any, next: any) => {
    if (req.url === "/dashboard" || req.url?.startsWith("/dashboard?")) {
      req.url = req.url.replace(/^\/dashboard/, "/dashboard.html");
    }
    next();
  };

  return {
    name: "thermaleye-dashboard-route",
    configureServer(server: any) {
      server.middlewares.use(rewriteDashboard);
    },
    configurePreviewServer(server: any) {
      server.middlewares.use(rewriteDashboard);
    },
  };
}

export default defineConfig({
  plugins: [react(), dashboardRoutePlugin()],
  build: {
    rollupOptions: {
      input: {
        main: resolve(rootDir, "index.html"),
        dashboard: resolve(rootDir, "dashboard.html"),
      },
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
      "/static": {
        target: "http://localhost:8000",
        changeOrigin: true,
      },
    },
  },
});
