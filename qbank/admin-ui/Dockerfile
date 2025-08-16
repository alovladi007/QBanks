FROM node:18-alpine

WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy application code
COPY . .

# Build the Next.js application
RUN npm run build

EXPOSE 4000

CMD ["npm", "start"]