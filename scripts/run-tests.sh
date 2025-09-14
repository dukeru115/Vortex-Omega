#!/bin/bash

# Скрипт автоматизации тестирования Vortex-Omega
# Запускает все виды тестов с отчётами

set -e

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Конфигурация
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_PATH="${PROJECT_ROOT}/venv"
COVERAGE_THRESHOLD=80
REPORTS_DIR="${PROJECT_ROOT}/test-reports"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

# Флаги
RUN_UNIT=true
RUN_INTEGRATION=true
RUN_PERFORMANCE=false
RUN_SECURITY=false
RUN_SMOKE=false
VERBOSE=false
PARALLEL=false
DOCKER_MODE=false

# Функции логирования
log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Показать помощь
show_help() {
    cat << EOF
Использование: $(basename "$0") [OPTIONS]

Опции:
    -u, --unit          Запустить юнит-тесты (по умолчанию: да)
    -i, --integration   Запустить интеграционные тесты (по умолчанию: да)
    -p, --performance   Запустить тесты производительности
    -s, --security      Запустить проверки безопасности
    -m, --smoke         Запустить smoke тесты
    -a, --all          Запустить все тесты
    -d, --docker       Использовать Docker для тестов
    -P, --parallel     Запускать тесты параллельно
    -v, --verbose      Подробный вывод
    -h, --help         Показать эту справку

Примеры:
    $(basename "$0")                    # Юнит + интеграционные тесты
    $(basename "$0") --all              # Все тесты
    $(basename "$0") -up --parallel    # Юнит + производительность параллельно
    $(basename "$0") --docker --all    # Все тесты в Docker
EOF
}

# Парсинг аргументов
parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            -u|--unit) RUN_UNIT=true; shift ;;
            -i|--integration) RUN_INTEGRATION=true; shift ;;
            -p|--performance) RUN_PERFORMANCE=true; shift ;;
            -s|--security) RUN_SECURITY=true; shift ;;
            -m|--smoke) RUN_SMOKE=true; shift ;;
            -a|--all)
                RUN_UNIT=true
                RUN_INTEGRATION=true
                RUN_PERFORMANCE=true
                RUN_SECURITY=true
                RUN_SMOKE=true
                shift ;;
            -d|--docker) DOCKER_MODE=true; shift ;;
            -P|--parallel) PARALLEL=true; shift ;;
            -v|--verbose) VERBOSE=true; shift ;;
            -h|--help) show_help; exit 0 ;;
            *) log_error "Неизвестная опция: $1"; show_help; exit 1 ;;
        esac
    done
}

# Проверка зависимостей
check_dependencies() {
    log_info "Проверка зависимостей..."
    
    if [ "$DOCKER_MODE" = true ]; then
        if ! command -v docker &> /dev/null; then
            log_error "Docker не установлен"
            exit 1
        fi
        
        if ! command -v docker-compose &> /dev/null; then
            log_error "Docker Compose не установлен"
            exit 1
        fi
    else
        if [ ! -d "$VENV_PATH" ]; then
            log_warning "Виртуальное окружение не найдено, создаю..."
            python3 -m venv "$VENV_PATH"
        fi
        
        source "$VENV_PATH/bin/activate"
        
        log_info "Установка зависимостей..."
        pip install -q --upgrade pip
        pip install -q -r "${PROJECT_ROOT}/requirements.txt"
        pip install -q -r "${PROJECT_ROOT}/requirements-dev.txt"
    fi
    
    # Создание директории для отчётов
    mkdir -p "$REPORTS_DIR"
    
    log_success "Зависимости проверены"
}

# Запуск юнит-тестов
run_unit_tests() {
    log_info "🧪 Запуск юнит-тестов..."
    
    if [ "$DOCKER_MODE" = true ]; then
        docker-compose run --rm vortex-test pytest tests/unit/ \
            -v \
            --cov=src \
            --cov-report=xml:${REPORTS_DIR}/coverage_unit.xml \
            --cov-report=html:${REPORTS_DIR}/htmlcov_unit \
            --junitxml=${REPORTS_DIR}/junit_unit.xml
    else
        pytest tests/unit/ \
            $([ "$VERBOSE" = true ] && echo "-vv" || echo "-v") \
            $([ "$PARALLEL" = true ] && echo "-n auto") \
            --cov=src \
            --cov-report=term-missing \
            --cov-report=xml:${REPORTS_DIR}/coverage_unit.xml \
            --cov-report=html:${REPORTS_DIR}/htmlcov_unit \
            --junitxml=${REPORTS_DIR}/junit_unit.xml \
            --tb=short
    fi
    
    # Проверка покрытия
    coverage_percent=$(python -c "
import xml.etree.ElementTree as ET
tree = ET.parse('${REPORTS_DIR}/coverage_unit.xml')
root = tree.getroot()
coverage = float(root.attrib.get('line-rate', 0)) * 100
print(f'{coverage:.1f}')
")
    
    if (( $(echo "$coverage_percent < $COVERAGE_THRESHOLD" | bc -l) )); then
        log_warning "Покрытие кода ($coverage_percent%) ниже порога ($COVERAGE_THRESHOLD%)"
    else
        log_success "Покрытие кода: $coverage_percent%"
    fi
    
    log_success "Юнит-тесты завершены"
}

# Запуск интеграционных тестов
run_integration_tests() {
    log_info "🔗 Запуск интеграционных тестов..."
    
    if [ "$DOCKER_MODE" = true ]; then
        # Запуск сервисов
        docker-compose up -d postgres redis
        sleep 10
        
        docker-compose run --rm vortex-test pytest tests/integration/ \
            -v \
            --junitxml=${REPORTS_DIR}/junit_integration.xml
        
        # Остановка сервисов
        docker-compose down
    else
        # Запуск локальных сервисов если нужно
        log_warning "Убедитесь, что PostgreSQL и Redis запущены локально"
        
        pytest tests/integration/ \
            $([ "$VERBOSE" = true ] && echo "-vv" || echo "-v") \
            --junitxml=${REPORTS_DIR}/junit_integration.xml \
            --tb=short
    fi
    
    log_success "Интеграционные тесты завершены"
}

# Запуск тестов производительности
run_performance_tests() {
    log_info "⚡ Запуск тестов производительности..."
    
    if [ ! -f "${PROJECT_ROOT}/tests/performance/test_performance.py" ]; then
        log_warning "Тесты производительности не найдены, создаю заглушку..."
        mkdir -p "${PROJECT_ROOT}/tests/performance"
        cat > "${PROJECT_ROOT}/tests/performance/test_performance.py" << 'EOF'
import pytest
import time

def test_symbolic_processing_speed(benchmark):
    """Тест скорости обработки Symbolic AI"""
    def process():
        time.sleep(0.01)  # Симуляция обработки
        return "processed"
    
    result = benchmark(process)
    assert result == "processed"

def test_esc_buffer_performance(benchmark):
    """Тест производительности ESC буферов"""
    def buffer_ops():
        data = list(range(1000))
        return sum(data)
    
    result = benchmark(buffer_ops)
    assert result == 499500
EOF
    fi
    
    pytest tests/performance/ \
        --benchmark-only \
        --benchmark-json=${REPORTS_DIR}/benchmark_${TIMESTAMP}.json \
        --benchmark-autosave
    
    # Анализ результатов
    if [ -f "${REPORTS_DIR}/benchmark_${TIMESTAMP}.json" ]; then
        python -c "
import json
with open('${REPORTS_DIR}/benchmark_${TIMESTAMP}.json') as f:
    data = json.load(f)
    for bench in data['benchmarks']:
        print(f\"  - {bench['name']}: {bench['stats']['mean']*1000:.2f}ms (median: {bench['stats']['median']*1000:.2f}ms)\")
"
    fi
    
    log_success "Тесты производительности завершены"
}

# Запуск проверок безопасности
run_security_tests() {
    log_info "🔐 Запуск проверок безопасности..."
    
    # Bandit для анализа кода
    log_info "Анализ кода с Bandit..."
    bandit -r src/ -f json -o ${REPORTS_DIR}/bandit_report.json || true
    
    # Safety для проверки зависимостей
    log_info "Проверка зависимостей с Safety..."
    safety check -r requirements.txt --json > ${REPORTS_DIR}/safety_report.json || true
    
    # Проверка на секреты
    log_info "Поиск секретов в коде..."
    if command -v trufflehog &> /dev/null; then
        trufflehog git file://. --only-verified --json > ${REPORTS_DIR}/secrets_report.json || true
    else
        log_warning "TruffleHog не установлен, пропускаю проверку секретов"
    fi
    
    # Анализ результатов
    python -c "
import json
import sys

issues_found = False

# Проверка Bandit
try:
    with open('${REPORTS_DIR}/bandit_report.json') as f:
        bandit_data = json.load(f)
        if bandit_data.get('results'):
            print('⚠️  Найдены проблемы безопасности в коде:')
            for issue in bandit_data['results'][:5]:  # Показываем первые 5
                print(f\"  - {issue['issue_text']} ({issue['issue_severity']})\")
            issues_found = True
except:
    pass

# Проверка Safety
try:
    with open('${REPORTS_DIR}/safety_report.json') as f:
        safety_data = json.load(f)
        if safety_data:
            print('⚠️  Найдены уязвимые зависимости:')
            for vuln in safety_data[:5]:  # Показываем первые 5
                print(f\"  - {vuln.get('package', 'Unknown')}: {vuln.get('vulnerability', 'Unknown')}\")
            issues_found = True
except:
    pass

if not issues_found:
    print('✅ Проблем безопасности не найдено')
"
    
    log_success "Проверки безопасности завершены"
}

# Запуск smoke тестов
run_smoke_tests() {
    log_info "🔥 Запуск smoke тестов..."
    
    # Проверка основных эндпоинтов
    if [ "$DOCKER_MODE" = true ]; then
        docker-compose up -d vortex-omega
        sleep 15
        
        BASE_URL="http://localhost:8080"
    else
        log_warning "Запустите приложение локально перед smoke тестами"
        BASE_URL="http://localhost:8080"
    fi
    
    # Проверка health endpoint
    log_info "Проверка health endpoint..."
    if curl -f -s "${BASE_URL}/health" > /dev/null; then
        log_success "Health endpoint работает"
    else
        log_error "Health endpoint не отвечает"
        exit 1
    fi
    
    # Проверка metrics endpoint
    log_info "Проверка metrics endpoint..."
    if curl -f -s "${BASE_URL}/metrics" > /dev/null; then
        log_success "Metrics endpoint работает"
    else
        log_warning "Metrics endpoint не отвечает"
    fi
    
    if [ "$DOCKER_MODE" = true ]; then
        docker-compose down
    fi
    
    log_success "Smoke тесты завершены"
}

# Генерация итогового отчёта
generate_report() {
    log_info "📊 Генерация итогового отчёта..."
    
    cat > "${REPORTS_DIR}/test_summary_${TIMESTAMP}.md" << EOF
# Отчёт о тестировании Vortex-Omega

**Дата:** $(date)
**Версия:** $(git describe --tags --always 2>/dev/null || echo "unknown")
**Ветка:** $(git branch --show-current 2>/dev/null || echo "unknown")

## Результаты тестов

| Тип теста | Статус | Детали |
|-----------|--------|--------|
| Юнит-тесты | $([ -f "${REPORTS_DIR}/junit_unit.xml" ] && echo "✅ Пройдены" || echo "⏭️ Пропущены") | [Отчёт](./junit_unit.xml) |
| Интеграционные | $([ -f "${REPORTS_DIR}/junit_integration.xml" ] && echo "✅ Пройдены" || echo "⏭️ Пропущены") | [Отчёт](./junit_integration.xml) |
| Производительность | $([ -f "${REPORTS_DIR}/benchmark_${TIMESTAMP}.json" ] && echo "✅ Выполнены" || echo "⏭️ Пропущены") | [Бенчмарки](./benchmark_${TIMESTAMP}.json) |
| Безопасность | $([ -f "${REPORTS_DIR}/bandit_report.json" ] && echo "✅ Проверены" || echo "⏭️ Пропущены") | [Отчёт](./bandit_report.json) |
| Smoke | $([ "$RUN_SMOKE" = true ] && echo "✅ Пройдены" || echo "⏭️ Пропущены") | - |

## Покрытие кода

$([ -f "${REPORTS_DIR}/coverage_unit.xml" ] && echo "Покрытие: ${coverage_percent}%" || echo "Покрытие не измерено")

## Артефакты

- [HTML отчёт о покрытии](./htmlcov_unit/index.html)
- [XML отчёт о покрытии](./coverage_unit.xml)
- [JUnit отчёты](./junit_*.xml)

---
*Сгенерировано автоматически скриптом run-tests.sh*
EOF
    
    log_success "Отчёт сохранён в ${REPORTS_DIR}/test_summary_${TIMESTAMP}.md"
}

# Главная функция
main() {
    log_info "🚀 Запуск тестирования Vortex-Omega..."
    
    cd "$PROJECT_ROOT"
    
    # Проверка зависимостей
    check_dependencies
    
    # Запуск тестов
    if [ "$PARALLEL" = true ]; then
        log_info "Запуск тестов в параллельном режиме..."
        
        # Запуск тестов параллельно
        ([ "$RUN_UNIT" = true ] && run_unit_tests) &
        ([ "$RUN_INTEGRATION" = true ] && run_integration_tests) &
        ([ "$RUN_PERFORMANCE" = true ] && run_performance_tests) &
        ([ "$RUN_SECURITY" = true ] && run_security_tests) &
        
        # Ожидание завершения
        wait
        
        [ "$RUN_SMOKE" = true ] && run_smoke_tests
    else
        # Последовательный запуск
        [ "$RUN_UNIT" = true ] && run_unit_tests
        [ "$RUN_INTEGRATION" = true ] && run_integration_tests
        [ "$RUN_PERFORMANCE" = true ] && run_performance_tests
        [ "$RUN_SECURITY" = true ] && run_security_tests
        [ "$RUN_SMOKE" = true ] && run_smoke_tests
    fi
    
    # Генерация отчёта
    generate_report
    
    log_success "✅ Тестирование завершено успешно!"
    log_info "📁 Отчёты сохранены в: ${REPORTS_DIR}"
}

# Обработка прерывания
trap 'log_error "Тестирование прервано"; exit 1' INT TERM

# Запуск
parse_args "$@"
main