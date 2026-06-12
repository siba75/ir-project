@echo off
title IR Project Services Launcher

echo =====================================
echo Starting IR Project Services...
echo =====================================

echo Starting Preprocessing Service on port 8001...
start "Preprocessing Service - 8001" cmd /k "cd /d D:\ir-project\services\preprocessing_service && D:\anaconda3\envs\ir-project\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8001"
timeout /t 3 /nobreak

echo Starting Indexing Service on port 8002...
start "Indexing Service - 8002" cmd /k "cd /d D:\ir-project\services\indexing_service && D:\anaconda3\envs\ir-project\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8002"
timeout /t 3 /nobreak

echo Starting Retrieval Service on port 8003...
start "Retrieval Service - 8003" cmd /k "cd /d D:\ir-project\services\retrieval_service && D:\anaconda3\envs\ir-project\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8003"
timeout /t 5 /nobreak

echo Starting Evaluation Service on port 8004...
start "Evaluation Service - 8004" cmd /k "cd /d D:\ir-project\services\evaluation_service && D:\anaconda3\envs\ir-project\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8004"
timeout /t 3 /nobreak

echo Starting Refinement Service on port 8005...
start "Refinement Service - 8005" cmd /k "cd /d D:\ir-project\services\refinement_service && D:\anaconda3\envs\ir-project\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8005"
timeout /t 5 /nobreak

echo Starting Gateway Service on port 8006...
start "Gateway Service - 8006" cmd /k "cd /d D:\ir-project\services\gateway_service && D:\anaconda3\envs\ir-project\python.exe -m uvicorn main:app --host 127.0.0.1 --port 8006"
timeout /t 5 /nobreak

echo Starting Streamlit UI on port 8501...
start "Streamlit UI - 8501" cmd /k "cd /d D:\ir-project\ui\streamlit_app && D:\anaconda3\envs\ir-project\python.exe -m streamlit run app.py --server.port 8501"

echo =====================================
echo All services are starting.
echo Open: http://localhost:8501
echo =====================================

pause
