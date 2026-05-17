@echo off
REM Usage: run_single.bat <image_path> [api_key] [project_id] [project_version] [results_dir]
REM Environment variables can also be used: ROBOFLOW_API_KEY, ROBOFLOW_PROJECT_ID, ROBOFLOW_PROJECT_VERSION, ROBOFLOW_RESULTS_DIR

set "API_KEY=%~2"
if "%API_KEY%"=="" set "API_KEY=%ROBOFLOW_API_KEY%"

set "PROJECT_ID=%~3"
if "%PROJECT_ID%"=="" set "PROJECT_ID=%ROBOFLOW_PROJECT_ID%"

set "PROJECT_VERSION=%~4"
if "%PROJECT_VERSION%"=="" set "PROJECT_VERSION=%ROBOFLOW_PROJECT_VERSION%"

set "RESULTS_DIR=%~5"
if "%RESULTS_DIR%"=="" set "RESULTS_DIR=%ROBOFLOW_RESULTS_DIR%"

if "%API_KEY%"=="" (
    echo Error: ROBOFLOW_API_KEY not set
    exit /b 1
)

python -m a2.annotate --image_path "%~1" --target_annotation_directory "%RESULTS_DIR%" --roboflow_project_id "%PROJECT_ID%" --roboflow_project_version %PROJECT_VERSION% --roboflow_api_key "%API_KEY%"
