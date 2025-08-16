#!/bin/bash

echo "🚀 QBank Setup Script"
echo "===================="

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check for Docker Compose
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    echo "Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "✅ Docker and Docker Compose found"

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
APP_SECRET=change-this-secret-in-production-$(openssl rand -hex 32)
NEXT_PUBLIC_API=http://localhost:8000
EOF
    echo "✅ .env file created"
fi

# Start services
echo "🐳 Starting Docker services..."
docker-compose up -d

# Wait for PostgreSQL to be ready
echo "⏳ Waiting for PostgreSQL to be ready..."
sleep 10

# Initialize database schema
echo "🗄️ Initializing database schema..."
docker-compose exec -T postgres psql -U qbank -d qbank < sql/content_ddl.sql
docker-compose exec -T postgres psql -U qbank -d qbank < sql/item_exposure_control.sql
docker-compose exec -T postgres psql -U qbank -d qbank < sql/feature_flags.sql
docker-compose exec -T postgres psql -U qbank -d qbank < sql/calibration_runs.sql

echo ""
echo "✅ QBank is now running!"
echo ""
echo "📍 Access points:"
echo "   - API:      http://localhost:8000"
echo "   - Admin UI: http://localhost:4000"
echo "   - API Docs: http://localhost:8000/docs"
echo ""
echo "🔑 To get an admin token, run:"
echo "   curl -X POST http://localhost:8000/v1/auth/mock-login \\"
echo "     -H 'Content-Type: application/json' \\"
echo "     -d '{\"user_id\":\"admin\",\"roles\":[\"admin\",\"author\",\"student\"]}'"
echo ""
echo "📊 To view logs:"
echo "   docker-compose logs -f"
echo ""
echo "🛑 To stop:"
echo "   docker-compose down"