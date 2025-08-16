# QBank Deployment Guide

## ⚠️ Important: This is NOT a Static Website

QBank is a **full-stack application** that requires:
- Backend API server (Python/FastAPI)
- PostgreSQL database
- Redis cache/queue
- Background worker processes
- Node.js server for the admin UI

**GitHub Pages cannot host this application** as it only serves static files.

## 🚀 Quick Start with Docker

### Prerequisites
- Docker Desktop installed ([Get Docker](https://docs.docker.com/get-docker/))
- Git installed
- 8GB RAM minimum
- Ports 4000, 5432, 6379, 8000 available

### One-Command Setup

```bash
# Clone the repository
git clone <your-repo-url> qbank
cd qbank

# Make setup script executable
chmod +x setup.sh

# Run setup (starts everything)
./setup.sh
```

This will:
1. Start PostgreSQL, Redis, API, Worker, and Admin UI
2. Initialize the database schema
3. Configure environment variables
4. Provide you with access URLs

### Access the Application

After setup completes:
- **Admin UI**: http://localhost:4000
- **API**: http://localhost:8000
- **API Documentation**: http://localhost:8000/docs

### Get Admin Access

```bash
# Get an admin JWT token
curl -X POST http://localhost:8000/v1/auth/mock-login \
  -H 'Content-Type: application/json' \
  -d '{"user_id":"admin","roles":["admin","author","student"]}'
```

Copy the `access_token` from the response and paste it in the Admin UI.

## 🌐 Cloud Deployment Options

### Option 1: Deploy to Heroku

```bash
# Install Heroku CLI
# Create Heroku app
heroku create your-qbank-app

# Add PostgreSQL and Redis
heroku addons:create heroku-postgresql:mini
heroku addons:create heroku-redis:mini

# Deploy
git push heroku main

# Run database migrations
heroku run psql $DATABASE_URL -f sql/content_ddl.sql
heroku run psql $DATABASE_URL -f sql/item_exposure_control.sql
heroku run psql $DATABASE_URL -f sql/feature_flags.sql
heroku run psql $DATABASE_URL -f sql/calibration_runs.sql
```

### Option 2: Deploy to AWS

Use AWS Elastic Beanstalk, ECS, or EC2:

1. **RDS** for PostgreSQL
2. **ElastiCache** for Redis
3. **Elastic Beanstalk** for the API
4. **ECS** for the worker
5. **Amplify** or **Vercel** for the Next.js UI

### Option 3: Deploy to Google Cloud

1. **Cloud SQL** for PostgreSQL
2. **Memorystore** for Redis
3. **Cloud Run** for the API and Worker
4. **Cloud Run** or **Firebase Hosting** for the UI

### Option 4: Deploy to DigitalOcean

```bash
# Use DigitalOcean App Platform
doctl apps create --spec app.yaml
```

Create `app.yaml`:
```yaml
name: qbank
services:
- name: api
  github:
    repo: your-username/qbank
    branch: main
  source_dir: qbank-backend
  dockerfile_path: Dockerfile
  http_port: 8000
  
- name: worker
  github:
    repo: your-username/qbank
    branch: main
  source_dir: qbank-backend
  dockerfile_path: Dockerfile
  run_command: python -m app.jobs.worker
  
- name: admin-ui
  github:
    repo: your-username/qbank
    branch: main
  source_dir: admin-ui
  dockerfile_path: Dockerfile
  http_port: 4000

databases:
- name: qbank-db
  engine: PG
  version: "16"
  
- name: qbank-redis
  engine: REDIS
  version: "7"
```

## 🔧 Manual Deployment (VPS/Dedicated Server)

### Requirements
- Ubuntu 22.04 or similar
- Python 3.11+
- Node.js 18+
- PostgreSQL 16
- Redis 7

### Steps

```bash
# 1. Install dependencies
sudo apt update
sudo apt install -y python3.11 python3-pip nodejs npm postgresql redis-server nginx

# 2. Clone repository
git clone <your-repo> /opt/qbank
cd /opt/qbank

# 3. Setup Python environment
python3 -m venv venv
source venv/bin/activate
pip install -r qbank-backend/requirements.txt

# 4. Setup database
sudo -u postgres createuser qbank
sudo -u postgres createdb qbank -O qbank
psql -U qbank -d qbank -f sql/content_ddl.sql
psql -U qbank -d qbank -f sql/item_exposure_control.sql
psql -U qbank -d qbank -f sql/feature_flags.sql
psql -U qbank -d qbank -f sql/calibration_runs.sql

# 5. Setup systemd services
# Create /etc/systemd/system/qbank-api.service
# Create /etc/systemd/system/qbank-worker.service

# 6. Setup Next.js UI
cd admin-ui
npm install
npm run build
pm2 start npm --name "qbank-ui" -- start

# 7. Configure Nginx reverse proxy
# Setup SSL with Let's Encrypt
```

## 📊 Production Considerations

### Environment Variables

Create a `.env` file:
```env
DATABASE_URL=postgresql://user:pass@host:5432/qbank
REDIS_URL=redis://host:6379/0
APP_SECRET=<generate-strong-secret>
NEXT_PUBLIC_API=https://api.your-domain.com
```

### Security
- Use HTTPS everywhere
- Set strong `APP_SECRET`
- Use environment-specific database credentials
- Enable CORS only for your domains
- Use rate limiting
- Implement proper logging

### Monitoring
- Use Sentry for error tracking
- Set up Prometheus/Grafana for metrics
- Configure health checks
- Set up backup strategies

### Scaling
- Use multiple API workers (Gunicorn)
- Scale Redis with Redis Cluster
- Use read replicas for PostgreSQL
- Implement CDN for static assets
- Use horizontal pod autoscaling in Kubernetes

## 🆘 Troubleshooting

### Database Connection Issues
```bash
# Check PostgreSQL is running
docker-compose ps
docker-compose logs postgres

# Test connection
docker-compose exec postgres psql -U qbank -d qbank -c "SELECT 1"
```

### Redis Connection Issues
```bash
# Check Redis is running
docker-compose logs redis

# Test connection
docker-compose exec redis redis-cli ping
```

### Port Already in Use
```bash
# Find and kill process using port
lsof -i :8000
kill -9 <PID>

# Or change ports in docker-compose.yml
```

### Admin UI Not Loading
- Check API is running: http://localhost:8000/health
- Check browser console for errors
- Verify NEXT_PUBLIC_API environment variable

## 📚 Additional Resources

- [FastAPI Deployment](https://fastapi.tiangolo.com/deployment/)
- [Next.js Deployment](https://nextjs.org/docs/deployment)
- [PostgreSQL Best Practices](https://wiki.postgresql.org/wiki/Main_Page)
- [Redis Best Practices](https://redis.io/docs/manual/patterns/)

## 💡 Need Help?

1. Check the logs: `docker-compose logs -f`
2. Review the [DEV.md](./DEV.md) file
3. Ensure all services are healthy: `docker-compose ps`
4. Verify database migrations ran successfully