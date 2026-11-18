# ===========================================================================
# ThermalEye — Frontend Multi-Stage Production Dockerfile
# ===========================================================================

# Stage 1: Build production SPA assets
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependency manifests and install cleanly
COPY package*.json ./
RUN npm ci

# Copy frontend source code and compile Vite bundle
COPY . .
RUN npm run build

# Stage 2: Serve optimized assets with Nginx
FROM nginx:alpine

# Remove default Nginx welcome page
RUN rm -rf /usr/share/nginx/html/*

# Copy compiled SPA bundle from builder stage
COPY --from=builder /app/dist /usr/share/nginx/html

# Copy custom reverse-proxy and SPA routing configuration
COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
