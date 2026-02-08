# Voyage Analytics - MLOps in Travel

A production-oriented Machine Learning + MLOps project that predicts prices and builds recommendation models for travel data, wrapped in a fully automated pipeline using modern tools and deployment standards.

## Project Overview

Voyage Analytics is a comprehensive ML system built to:
- Predict flight and hotel prices
- Perform user segmentation and gender classification
- Recommend suitable hotels to users

This project showcases an end-to-end data science workflow—from data handling and modeling to deployment and monitoring—using MLOps best practices.

## Architecture & Workflow

This project applies an MLOps-style production pipeline:

Data (via DVC) → Tracking (MLflow) → Model APIs (FastAPI) → Containerization (Docker) → Deployment + Monitoring → User UI (Streamlit)

## Tech Stack

- Python & Jupyter Notebooks
- Data Version Control (DVC)
- MLflow for experiment tracking
- FastAPI for serving model endpoints
- Docker & Kubernetes
- Streamlit for interactive user interface
-CI/CD (optional / extensible)

Tools and processes combine data science workflows with ML production readiness

## Project Structure
├── .dvc
├── api/
├── app/
├── config/
├── data/
├── mlartifacts/
├── mlruns/
├── notebooks/
├── presentation/
├── src/
├── Dockerfile.api
├── Dockerfile.streamlit
├── docker-compose.yml
├── requirements.txt
└── README.md

## Quick Setup

1. Clone the repo:
git clone https://github.com/SrushtiSuthar/Voyage_Analytics_MLOps.git
cd Voyage_Analytics_MLOps

2. Create and activate environment:
conda create -n voyage_ml python=3.9
conda activate voyage_ml

3. Install dependencies:
pip install -r requirements.txt

4. Pull data & models via DVC:
dvc pull

5. Launch MLflow:
mlflow ui --port 5000

6. Start Jupyter Lab:
jupyter lab --notebook-dir=notebooks

## What You’ll Learn

This project demonstrates:

- Structuring modular ML systems
- Versioning data and models with DVC
- Tracking experiments with MLflow
- Serving models via FastAPI
- Deploying containerized services with Docker
- Creating interactive dashboards using Streamlit

## Future Improvements

Possible enhancements:

- Add automated CI/CD pipelines
- Expand model evaluation tracking
- Build deeper user behavior insights
- Deploy models in cloud environments

## Credits

Developed by Srushti Suthar as part of a Data Science internship project showcasing real-world ML + MLOps skills.