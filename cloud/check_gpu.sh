#!/bin/bash
# ============================================================================
# check_gpu.sh — GPU & System Healthcheck for LLM Training VM
# ============================================================================
# Usage: bash check_gpu.sh
# Returns: 0 if all checks pass, 1 if any check fails
# ============================================================================

set -euo pipefail

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'
BOLD='\033[1m'

PASS=0
FAIL=0

check() {
    local name="$1"
    local status="$2"
    local detail="$3"
    if [ "$status" -eq 0 ]; then
        echo -e "  ${GREEN}[OK]${NC}  ${name}: ${detail}"
        ((PASS++))
    else
        echo -e "  ${RED}[FAIL]${NC} ${name}: ${detail}"
        ((FAIL++))
    fi
}

echo -e "\n${BOLD}${CYAN}========================================${NC}"
echo -e "${BOLD}${CYAN}  GPU & System Healthcheck${NC}"
echo -e "${BOLD}${CYAN}========================================${NC}\n"

# 1. NVIDIA Driver
echo -e "${BOLD}--- NVIDIA Driver ---${NC}"
if command -v nvidia-smi &>/dev/null; then
    DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    check "NVIDIA Driver" 0 "v${DRIVER}"
else
    check "NVIDIA Driver" 1 "nvidia-smi not found"
fi

# 2. CUDA Version
echo -e "${BOLD}--- CUDA ---${NC}"
if command -v nvidia-smi &>/dev/null; then
    CUDA=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
    CUDA_VER=$(nvidia-smi 2>/dev/null | grep "CUDA Version" | awk '{print $9}')
    if [ -n "$CUDA_VER" ]; then
        check "CUDA Version" 0 "${CUDA_VER}"
    else
        check "CUDA Version" 1 "Could not detect CUDA version"
    fi
else
    check "CUDA Version" 1 "nvidia-smi not available"
fi

# 3. GPU Model & Memory
echo -e "${BOLD}--- GPU ---${NC}"
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    if echo "$GPU_NAME" | grep -qi "A100\|H100\|RTX"; then
        check "GPU Model" 0 "${GPU_NAME} (${GPU_MEM})"
    else
        check "GPU Model" 1 "Unexpected GPU: ${GPU_NAME} (${GPU_MEM})"
    fi
else
    check "GPU Model" 1 "No GPU detected"
fi

# 4. GPU Temperature
echo -e "${BOLD}--- GPU Temperature ---${NC}"
if command -v nvidia-smi &>/dev/null; then
    TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader 2>/dev/null | head -1)
    if [ "$TEMP" -lt 85 ] 2>/dev/null; then
        check "GPU Temp" 0 "${TEMP}°C"
    else
        check "GPU Temp" 1 "${TEMP}°C (too hot!)"
    fi
else
    check "GPU Temp" 1 "Cannot read temperature"
fi

# 5. GPU Utilization
echo -e "${BOLD}--- GPU Utilization ---${NC}"
if command -v nvidia-smi &>/dev/null; then
    UTIL=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | head -1)
    MEM_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | head -1)
    MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null | head -1)
    check "GPU Utilization" 0 "${UTIL} | Memory: ${MEM_USED} / ${MEM_TOTAL}"
else
    check "GPU Utilization" 1 "Cannot query GPU"
fi

# 6. PyTorch CUDA
echo -e "${BOLD}--- PyTorch ---${NC}"
if command -v python3 &>/dev/null || command -v python &>/dev/null; then
    PY_CMD=$(command -v python3 || command -v python)
    TORCH_CUDA=$($PY_CMD -c "import torch; print(torch.cuda.is_available())" 2>/dev/null)
    if [ "$TORCH_CUDA" = "True" ]; then
        TORCH_DEV=$($PY_CMD -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
        check "PyTorch CUDA" 0 "Available (${TORCH_DEV})"
    else
        check "PyTorch CUDA" 1 "CUDA not available in PyTorch"
    fi
else
    check "PyTorch CUDA" 1 "Python not found"
fi

# 7. PyTorch Version
if command -v python3 &>/dev/null || command -v python &>/dev/null; then
    PY_CMD=$(command -v python3 || command -v python)
    TORCH_VER=$($PY_CMD -c "import torch; print(torch.__version__)" 2>/dev/null)
    if [ -n "$TORCH_VER" ]; then
        check "PyTorch Version" 0 "${TORCH_VER}"
    else
        check "PyTorch Version" 1 "PyTorch not installed"
    fi
fi

# 8. Disk Space
echo -e "${BOLD}--- Disk ---${NC}"
DISK_AVAIL=$(df -h / 2>/dev/null | awk 'NR==2 {print $4}')
DISK_TOTAL=$(df -h / 2>/dev/null | awk 'NR==2 {print $2}')
DISK_PCT=$(df / 2>/dev/null | awk 'NR==2 {print $5}' | tr -d '%')
if [ "${DISK_PCT:-100}" -lt 90 ] 2>/dev/null; then
    check "Disk Space" 0 "${DISK_AVAIL} free / ${DISK_TOTAL} total (${DISK_PCT}% used)"
else
    check "Disk Space" 1 "${DISK_AVAIL} free / ${DISK_TOTAL} total (${DISK_PCT}% used) — LOW!"
fi

# 9. RAM
echo -e "${BOLD}--- Memory ---${NC}"
if command -v free &>/dev/null; then
    RAM_TOTAL=$(free -h 2>/dev/null | awk '/^Mem:/ {print $2}')
    RAM_AVAIL=$(free -h 2>/dev/null | awk '/^Mem:/ {print $7}')
    RAM_PCT=$(free 2>/dev/null | awk '/^Mem:/ {printf "%.0f", ($2-$7)/$2*100}')
    if [ "${RAM_PCT:-100}" -lt 90 ] 2>/dev/null; then
        check "RAM" 0 "${RAM_AVAIL} available / ${RAM_TOTAL} total (${RAM_PCT}% used)"
    else
        check "RAM" 1 "${RAM_AVAIL} available / ${RAM_TOTAL} total (${RAM_PCT}% used) — LOW!"
    fi
else
    check "RAM" 1 "free command not available"
fi

# 10. Python Version
echo -e "${BOLD}--- Python ---${NC}"
if command -v python3 &>/dev/null || command -v python &>/dev/null; then
    PY_CMD=$(command -v python3 || command -v python)
    PY_VER=$($PY_CMD --version 2>&1)
    check "Python" 0 "${PY_VER}"
else
    check "Python" 1 "Python not found"
fi

# Summary
echo -e "\n${BOLD}${CYAN}========================================${NC}"
echo -e "  ${GREEN}Passed: ${PASS}${NC}  |  ${RED}Failed: ${FAIL}${NC}"
echo -e "${BOLD}${CYAN}========================================${NC}\n"

if [ "$FAIL" -gt 0 ]; then
    echo -e "${RED}${BOLD}  HEALTHCHECK FAILED${NC}\n"
    exit 1
else
    echo -e "${GREEN}${BOLD}  ALL CHECKS PASSED${NC}\n"
    exit 0
fi
