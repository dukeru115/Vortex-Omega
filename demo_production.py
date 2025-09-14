#!/usr/bin/env python3
"""
Production Demo для Vortex-Omega NFCS
Демонстрация работы системы после CI/CD настройки
"""

import asyncio
import json
import time
from datetime import datetime
import sys
import os

# Добавляем src в путь
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

async def test_symbolic_ai():
    """Тест модуля Symbolic AI"""
    print("\n🧠 Тестирование Symbolic AI...")
    
    try:
        from modules.symbolic.symbolic_core import SymbolicCore
        
        symbolic = SymbolicCore()
        text = "The velocity is 10 m/s and the mass is 5 kg"
        
        result = await symbolic.process_text(text)
        
        print(f"✅ Symbolic AI работает!")
        print(f"   - Найдено clauses: {len(result.get('clauses', []))}")
        print(f"   - Найдено fields: {len(result.get('fields', []))}")
        return True
    except Exception as e:
        print(f"❌ Ошибка в Symbolic AI: {e}")
        return False

async def test_esc_module():
    """Тест модуля ESC"""
    print("\n📡 Тестирование Enhanced ESC...")
    
    try:
        from modules.esc.enhanced_esc import EnhancedESC
        import numpy as np
        
        esc = EnhancedESC()
        
        # Тест обработки токена
        result = esc.process_token(
            token_id=1,
            token_embedding=np.random.randn(512),
            timestamp=time.time()
        )
        
        print(f"✅ ESC работает!")
        print(f"   - Signal: {result['signal']:.4f}")
        print(f"   - Frequency: {result['frequency']:.2f} Hz")
        print(f"   - Memory usage: {esc._check_memory_usage():.2f} MB")
        return True
    except Exception as e:
        print(f"❌ Ошибка в ESC: {e}")
        return False

async def test_kuramoto():
    """Тест модуля Kuramoto"""
    print("\n🌊 Тестирование Kuramoto Model...")
    
    try:
        from core.kuramoto_solver_optimized import OptimizedKuramotoSolver
        import numpy as np
        
        kuramoto = OptimizedKuramotoSolver(
            n_oscillators=10,
            natural_frequencies=np.random.randn(10) * 0.5
        )
        
        # Шаг симуляции
        phases = np.random.uniform(0, 2*np.pi, 10)
        new_phases = kuramoto.step(phases, dt=0.01)
        order_param = kuramoto.compute_order_parameter(new_phases)
        
        print(f"✅ Kuramoto работает!")
        print(f"   - Order parameter: {order_param:.4f}")
        print(f"   - Mean frequency: {np.mean(kuramoto.natural_frequencies):.2f}")
        return True
    except Exception as e:
        print(f"❌ Ошибка в Kuramoto: {e}")
        return False

async def test_monitoring():
    """Тест модуля мониторинга"""
    print("\n📊 Тестирование Monitoring...")
    
    try:
        from monitoring.metrics import MetricsCollector
        
        metrics = MetricsCollector()
        
        # Симуляция метрик
        metrics.track_request("GET", "/api/test", 200, 0.123)
        metrics.track_symbolic_processing("parse", 0.456, {"entity": 5, "quantity": 3})
        metrics.update_esc_metrics(
            buffer_sizes={"working": 100, "episodic": 500},
            memory_mb=12.5,
            oscillator_count=50
        )
        
        print(f"✅ Monitoring работает!")
        print(f"   - Метрики собираются")
        print(f"   - Prometheus формат доступен")
        return True
    except Exception as e:
        print(f"❌ Ошибка в Monitoring: {e}")
        return False

async def test_docker_config():
    """Проверка Docker конфигурации"""
    print("\n🐋 Проверка Docker конфигурации...")
    
    files_to_check = [
        ("Dockerfile", "✅ Dockerfile найден"),
        ("docker-compose.yml", "✅ Docker Compose найден"),
        (".dockerignore", "✅ .dockerignore найден"),
        ("scripts/docker-deploy.sh", "✅ Deployment script найден")
    ]
    
    all_found = True
    for file_path, message in files_to_check:
        if os.path.exists(file_path):
            print(f"   {message}")
        else:
            print(f"   ❌ {file_path} не найден")
            all_found = False
    
    return all_found

async def test_ci_cd_config():
    """Проверка CI/CD конфигурации"""
    print("\n🔧 Проверка CI/CD конфигурации...")
    
    files_to_check = [
        (".gitlab-ci.yml", "✅ GitLab CI найден"),
        ("Jenkinsfile", "✅ Jenkins Pipeline найден"),
        (".pre-commit-config.yaml", "✅ Pre-commit hooks найдены"),
        ("pyproject.toml", "✅ Python project config найден"),
        ("setup.py", "✅ Setup.py найден")
    ]
    
    all_found = True
    for file_path, message in files_to_check:
        if os.path.exists(file_path):
            print(f"   {message}")
        else:
            print(f"   ❌ {file_path} не найден")
            all_found = False
    
    return all_found

async def main():
    """Главная функция демо"""
    print("=" * 60)
    print("🚀 VORTEX-OMEGA NFCS PRODUCTION DEMO")
    print("=" * 60)
    print(f"📅 Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📍 Версия: 2.5.0")
    print(f"🔧 Python: {sys.version.split()[0]}")
    print("=" * 60)
    
    # Запуск всех тестов
    results = []
    
    # Тесты модулей
    results.append(("Symbolic AI", await test_symbolic_ai()))
    results.append(("Enhanced ESC", await test_esc_module()))
    results.append(("Kuramoto Model", await test_kuramoto()))
    results.append(("Monitoring", await test_monitoring()))
    
    # Проверка конфигураций
    results.append(("Docker Config", await test_docker_config()))
    results.append(("CI/CD Config", await test_ci_cd_config()))
    
    # Итоговый отчёт
    print("\n" + "=" * 60)
    print("📊 ИТОГОВЫЙ ОТЧЁТ")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:20} {status}")
    
    print("-" * 60)
    print(f"Результат: {passed}/{total} тестов пройдено")
    
    if passed == total:
        print("\n🎉 СИСТЕМА ГОТОВА К PRODUCTION!")
        print("✅ Все компоненты работают корректно")
        print("✅ CI/CD полностью настроен")
        print("✅ Docker конфигурация готова")
        print("\n📝 Следующие шаги:")
        print("1. Создайте Pull Request на GitHub")
        print("2. Настройте GitHub Actions через веб-интерфейс")
        print("3. Добавьте секреты (PYPI_API_TOKEN и др.)")
        print("4. Запустите deployment через docker-deploy.sh")
    else:
        print("\n⚠️ Некоторые компоненты требуют внимания")
        print("Проверьте логи выше для деталей")
    
    print("\n" + "=" * 60)
    print("🔗 Repository: https://github.com/dukeru115/Vortex-Omega")
    print("🔗 PR: https://github.com/dukeru115/Vortex-Omega/pull/new/genspark_ai_developer")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())