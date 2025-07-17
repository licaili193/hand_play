# PowerShell script to setup PianoMotion10M repository
# Run this script once to clone the repo and install dependencies

Write-Host "Setting up PianoMotion10M repository..." -ForegroundColor Green

# Check if git is available
try {
    git --version | Out-Null
    Write-Host "Git is available" -ForegroundColor Green
} catch {
    Write-Host "Error: Git is not installed or not in PATH" -ForegroundColor Red
    Write-Host "Please install Git from https://git-scm.com/" -ForegroundColor Yellow
    exit 1
}

# Set repo directory
$repoDir = "PianoMotion10M"
if (Test-Path $repoDir) {
    Write-Host "Repository directory already exists. Removing..." -ForegroundColor Yellow
    Remove-Item -Recurse -Force $repoDir
}

# Clone the repository
Write-Host "Cloning PianoMotion10M repository..." -ForegroundColor Green
try {
    git clone --depth 1 https://github.com/agnJason/PianoMotion10M.git $repoDir
    Write-Host "Repository cloned successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error cloning repository: $_" -ForegroundColor Red
    exit 1
}

# Install Python dependencies
Write-Host "Installing Python dependencies..." -ForegroundColor Green
try {
    python -m pip install librosa soundfile tqdm
    Write-Host "Dependencies installed successfully!" -ForegroundColor Green
} catch {
    Write-Host "Error installing dependencies: $_" -ForegroundColor Red
    Write-Host "You may need to install them manually: pip install librosa soundfile tqdm" -ForegroundColor Yellow
}

# Optional: Install PyTorch for GPU rendering (if you have a GPU)
Write-Host "Do you want to install PyTorch for GPU rendering? (y/n)" -ForegroundColor Yellow
$response = Read-Host
if ($response -eq "y" -or $response -eq "Y") {
    Write-Host "Installing PyTorch..." -ForegroundColor Green
    try {
        python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        Write-Host "PyTorch installed successfully!" -ForegroundColor Green
    } catch {
        Write-Host "Error installing PyTorch: $_" -ForegroundColor Red
        Write-Host "You can install it manually later if needed" -ForegroundColor Yellow
    }
}

Write-Host "Setup complete!" -ForegroundColor Green
Write-Host "The repository is now available in the '$repoDir' directory" -ForegroundColor Cyan
Write-Host "You can now run your MIDI to frames script without downloading the repo each time." -ForegroundColor Cyan 