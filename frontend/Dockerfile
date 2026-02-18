FROM node:20-slim

WORKDIR /app

# Copy dependency files first (layer caching)
COPY package.json package-lock.json* ./

# Install dependencies
RUN npm install

# Copy source (will be overridden by volume mount in dev)
COPY . .

EXPOSE 5173

CMD ["npm", "run", "dev", "--", "--host", "0.0.0.0"]
