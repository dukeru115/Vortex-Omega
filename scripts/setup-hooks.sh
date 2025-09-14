#!/bin/bash

# Скрипт установки pre-commit hooks для Vortex-Omega

set -e

# Цвета
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🔧 Установка pre-commit hooks для Vortex-Omega${NC}"

# Проверка Python
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python 3 не установлен${NC}"
    exit 1
fi

# Установка pre-commit
echo -e "${YELLOW}📦 Установка pre-commit...${NC}"
pip install --user pre-commit

# Установка хуков
echo -e "${YELLOW}🔗 Установка git hooks...${NC}"
pre-commit install
pre-commit install --hook-type commit-msg
pre-commit install --hook-type pre-push

# Создание baseline для detect-secrets
echo -e "${YELLOW}🔐 Создание baseline для detect-secrets...${NC}"
detect-secrets scan --baseline .secrets.baseline || true

# Первый запуск на всех файлах (опционально)
read -p "Запустить проверку на всех файлах? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}🏃 Запуск проверок...${NC}"
    pre-commit run --all-files || true
fi

echo -e "${GREEN}✅ Pre-commit hooks установлены!${NC}"
echo -e "${GREEN}Теперь перед каждым коммитом будут выполняться автоматические проверки.${NC}"