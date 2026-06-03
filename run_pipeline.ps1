
# run_pipeline.ps1
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

Write-Host "Creando entorno virtual..." -ForegroundColor Cyan
python -m venv venv

Write-Host "Activando entorno virtual..." -ForegroundColor Cyan
.\venv\Scripts\Activate.ps1

Write-Host "Instalando dependencias..." -ForegroundColor Cyan
pip install -r requirements.txt

Write-Host "Ejecutando captura de caras..." -ForegroundColor Cyan
python captura-caras.py

Write-Host "Ejecutando el entrenamiento..." -ForegroundColor Cyan
python entrenamiento.py

Write-Host "Ejecutando prueba de reconocimiento..." -ForegroundColor Cyan
python reconoce.py

Write-Host "Ejecución del pipeline completada." -ForegroundColor Green