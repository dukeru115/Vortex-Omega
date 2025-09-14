#!/bin/bash

# Quick Start Production Script для Vortex-Omega
# Быстрый запуск системы в production

set -e

echo "🚀 VORTEX-OMEGA QUICK START PRODUCTION"
echo "======================================"
echo ""

# Проверка Docker
if ! command -v docker &> /dev/null; then
    echo "⚠️  Docker не установлен. Установка..."
    echo ""
    echo "Для Ubuntu/Debian:"
    echo "  sudo apt-get update"
    echo "  sudo apt-get install docker.io docker-compose"
    echo ""
    echo "Для macOS:"
    echo "  Установите Docker Desktop с https://www.docker.com/products/docker-desktop"
    echo ""
    exit 1
fi

# Меню выбора
echo "Выберите действие:"
echo "1) 🐋 Запустить через Docker Compose"
echo "2) 🐍 Запустить локально (Python)"
echo "3) 🧪 Запустить тесты"
echo "4) 📦 Собрать Python пакет"
echo "5) 🔧 Установить pre-commit hooks"
echo ""
read -p "Ваш выбор (1-5): " choice

case $choice in
    1)
        echo "🐋 Запуск через Docker Compose..."
        docker-compose up -d
        echo "✅ Сервисы запущены!"
        echo "📊 Проверьте статус: docker-compose ps"
        echo "📝 Логи: docker-compose logs -f"
        echo "🌐 Приложение: http://localhost:8080"
        echo "📈 Grafana: http://localhost:3000 (admin/vortex123)"
        ;;
    
    2)
        echo "🐍 Запуск локально..."
        if [ ! -d "venv" ]; then
            echo "Создание виртуального окружения..."
            python3 -m venv venv
        fi
        source venv/bin/activate
        pip install -r requirements.txt
        echo "✅ Зависимости установлены!"
        echo ""
        echo "Запустите сервер:"
        echo "  python src/api/server.py"
        ;;
    
    3)
        echo "🧪 Запуск тестов..."
        ./scripts/run-tests.sh --all
        ;;
    
    4)
        echo "📦 Сборка пакета..."
        python3 -m pip install --upgrade build
        python3 -m build
        echo "✅ Пакет собран в директории dist/"
        ;;
    
    5)
        echo "🔧 Установка pre-commit hooks..."
        pip install pre-commit
        pre-commit install
        echo "✅ Pre-commit hooks установлены!"
        ;;
    
    *)
        echo "❌ Неверный выбор"
        exit 1
        ;;
esac

echo ""
echo "======================================"
echo "📚 Документация: README.md"
echo "🔗 GitHub: https://github.com/dukeru115/Vortex-Omega"
echo "======================================"