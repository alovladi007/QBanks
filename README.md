# QBank Enterprise - Advanced Question Bank Management System

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115.0-green.svg)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Next.js-14.1.0-black.svg)](https://nextjs.org/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue.svg)](https://www.docker.com/)

## 🚀 Overview

QBank Enterprise is a state-of-the-art, commercial-grade Question Bank Management System designed for educational institutions, certification bodies, and corporate training departments. Built with cutting-edge technology and AI-powered features, it provides a comprehensive solution for creating, managing, and delivering assessments at scale.

## ✨ Key Features

### 📝 Question Management
- **Multi-format Support**: MCQ, True/False, Short Answer, Essay, Fill-in-the-blanks, Matching, and more
- **Version Control**: Complete version history with rollback capabilities
- **Rich Media Support**: Images, videos, audio, LaTeX math, code snippets
- **Question Pools**: Organize questions into hierarchical topics and subtopics
- **Metadata & Tagging**: Comprehensive tagging system with auto-tagging
- **Import/Export**: Support for QTI, GIFT, Moodle XML, Excel, CSV formats

### 🤖 AI-Powered Features
- **Intelligent Question Generation**: AI creates questions based on learning objectives
- **Auto-tagging & Classification**: Automatic topic and difficulty assignment
- **Plagiarism Detection**: Check for duplicate or similar questions
- **Quality Assessment**: AI evaluates question clarity and effectiveness
- **Content Enhancement**: Improve question wording and distractors
- **Adaptive Testing**: IRT-based computerized adaptive testing

### 📊 Analytics & Reporting
- **Real-time Analytics Dashboard**: Track performance metrics live
- **Item Response Theory (IRT)**: Advanced psychometric analysis
- **Difficulty Calibration**: Automatic difficulty estimation
- **Discrimination Analysis**: Identify questions that differentiate ability levels
- **Student Performance Tracking**: Individual and cohort analytics
- **Custom Reports**: Generate detailed PDF/Excel reports

### 👥 User Management
- **Multi-tenancy**: Support for multiple organizations
- **Role-Based Access Control (RBAC)**: Granular permission system
- **Single Sign-On (SSO)**: OAuth2 with Google, Microsoft, GitHub
- **Two-Factor Authentication (2FA)**: Enhanced security
- **API Key Management**: Programmatic access for integrations
- **Audit Trail**: Complete activity logging

### 💳 Commercial Features
- **Subscription Management**: Tiered pricing (Free, Basic, Premium, Enterprise)
- **Payment Integration**: Stripe and PayPal support
- **Usage Limits**: Question, test, and student limits per tier
- **White-labeling**: Custom branding for enterprise clients
- **SLA Monitoring**: Service level agreement tracking

### 🔒 Security & Compliance
- **End-to-end Encryption**: Data encrypted at rest and in transit
- **GDPR Compliant**: Full data privacy compliance
- **Regular Security Audits**: Automated vulnerability scanning
- **Backup & Recovery**: Automated backups with point-in-time recovery
- **Rate Limiting**: API rate limiting and DDoS protection

### 🚀 Performance & Scalability
- **Microservices Architecture**: Scalable and maintainable
- **Caching Layer**: Redis for high-performance caching
- **CDN Integration**: Global content delivery
- **Load Balancing**: Horizontal scaling support
- **Real-time Updates**: WebSocket support for live features

## 🛠️ Technology Stack

### Backend
- **Framework**: FastAPI (Python 3.11+)
- **Database**: PostgreSQL with pgvector extension
- **Cache**: Redis
- **Search**: Elasticsearch
- **Queue**: Celery with RabbitMQ
- **Storage**: MinIO/S3
- **ML/AI**: PyTorch, Transformers, OpenAI API

### Frontend
- **Framework**: Next.js 14 with TypeScript
- **UI Library**: Radix UI with Tailwind CSS
- **State Management**: Zustand
- **Data Fetching**: TanStack Query
- **Forms**: React Hook Form with Zod validation
- **Charts**: Recharts & Chart.js
- **Editor**: TipTap with math support

### Infrastructure
- **Containerization**: Docker & Docker Compose
- **Orchestration**: Kubernetes ready
- **Monitoring**: Prometheus & Grafana
- **Logging**: ELK Stack
- **CI/CD**: GitHub Actions
- **Cloud**: AWS/GCP/Azure compatible

## 📦 Installation

### Prerequisites
- Docker & Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)
- PostgreSQL 15+ (for local development)

### Quick Start with Docker

1. **Clone the repository**
```bash
git clone https://github.com/your-org/qbank-enterprise.git
cd qbank-enterprise
```

2. **Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. **Start the application**
```bash
docker-compose up -d
```

4. **Access the application**
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Documentation: http://localhost:8000/docs
- Grafana: http://localhost:3001 (admin/admin)
- Flower (Celery): http://localhost:5555

### Local Development Setup

#### Backend Setup
```bash
cd backend
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
alembic upgrade head
uvicorn app.main:app --reload
```

#### Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## 🔧 Configuration

### Environment Variables

Key environment variables:

```env
# Application
ENVIRONMENT=development
SECRET_KEY=your-secret-key

# Database
DATABASE_URL=postgresql://user:pass@localhost/qbank
REDIS_URL=redis://localhost:6379

# AI/ML
OPENAI_API_KEY=your-openai-key
ANTHROPIC_API_KEY=your-anthropic-key

# Payment
STRIPE_SECRET_KEY=your-stripe-key
STRIPE_PUBLISHABLE_KEY=your-stripe-pub-key

# Storage
S3_ACCESS_KEY_ID=your-access-key
S3_SECRET_ACCESS_KEY=your-secret-key
S3_BUCKET_NAME=qbank-assets

# Email
SMTP_HOST=smtp.gmail.com
SMTP_USER=your-email
SMTP_PASSWORD=your-password
```

## 📚 API Documentation

### Authentication
```bash
# Login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email": "user@example.com", "password": "password"}'

# Use token
curl -X GET http://localhost:8000/api/v1/questions \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Question Generation
```python
import requests

response = requests.post(
    "http://localhost:8000/api/v1/questions/generate",
    headers={"Authorization": f"Bearer {token}"},
    json={
        "topic": "Python Programming",
        "learning_objectives": ["Understand loops", "Apply functions"],
        "difficulty_level": "Medium",
        "question_types": ["MCQ", "ShortAnswer"],
        "count": 10
    }
)
questions = response.json()
```

## 🧪 Testing

### Backend Tests
```bash
cd backend
pytest tests/ -v --cov=app
```

### Frontend Tests
```bash
cd frontend
npm run test
npm run test:e2e
```

### Load Testing
```bash
locust -f tests/load/locustfile.py --host=http://localhost:8000
```

## 📈 Monitoring

### Metrics
- Prometheus metrics available at `/metrics`
- Grafana dashboards for visualization
- Custom business metrics tracking

### Logging
- Structured JSON logging
- Centralized log aggregation
- Real-time log streaming

### Health Checks
- `/health` - Basic health check
- `/health/ready` - Readiness probe
- `/health/live` - Liveness probe

## 🚀 Deployment

### Production Deployment

#### Using Docker Compose
```bash
docker-compose -f docker-compose.prod.yml up -d
```

#### Kubernetes Deployment
```bash
kubectl apply -f k8s/
```

#### Cloud Deployment
- **AWS**: Use ECS/EKS with RDS and ElastiCache
- **GCP**: Use Cloud Run/GKE with Cloud SQL
- **Azure**: Use Container Instances/AKS with Azure Database

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [https://docs.qbank.com](https://docs.qbank.com)
- **Email**: support@qbank.com
- **Discord**: [Join our community](https://discord.gg/qbank)
- **Issues**: [GitHub Issues](https://github.com/your-org/qbank-enterprise/issues)

## 🏆 Enterprise Support

For enterprise support, custom features, and SLA guarantees:
- Contact: enterprise@qbank.com
- Phone: +1-xxx-xxx-xxxx

## 🎯 Roadmap

### Q1 2024
- [ ] Mobile applications (iOS/Android)
- [ ] Advanced proctoring features
- [ ] Blockchain-based certificates

### Q2 2024
- [ ] Virtual classroom integration
- [ ] Advanced analytics with ML insights
- [ ] Multi-language support (10+ languages)

### Q3 2024
- [ ] AR/VR question types
- [ ] Voice-based assessments
- [ ] Advanced anti-cheating mechanisms

## 🙏 Acknowledgments

- FastAPI for the amazing framework
- Next.js team for the frontend framework
- All our contributors and users

---

**Built with ❤️ by the QBank Team**