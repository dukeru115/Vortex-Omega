# Участие в разработке NFCS

Мы приветствуем участие сообщества! Вот как вы можете внести свой вклад:

## 🐛 Сообщение об ошибках

1. Проверьте [Issues](https://github.com/dukeru115/Vortex-Omega/issues) на предмет существующих отчетов
2. Создайте новый Issue с подробным описанием проблемы
3. Приложите минимальный воспроизводимый пример

## 💡 Предложение новых функций

1. Откройте Issue с тегом "enhancement"
2. Опишите предлагаемую функциональность
3. Обоснуйте ее полезность для проекта

## 🔧 Отправка Pull Request

1. **Fork** репозитория
2. Создайте feature branch: `git checkout -b feature/amazing-feature`
3. Внесите изменения и добавьте тесты
4. Убедитесь, что все тесты проходят: `pytest`
5. Зафиксируйте изменения: `git commit -m 'Add amazing feature'`
6. Push в branch: `git push origin feature/amazing-feature`
7. Откройте Pull Request

## 📋 Стандарты разработки

- **Стиль кода**: PEP 8 (используйте `black` для форматирования)
- **Документация**: Все публичные функции должны иметь docstrings
- **Тесты**: Новый код должен покрываться тестами
- **Коммиты**: Используйте [Conventional Commits](https://www.conventionalcommits.org/)

## 🧪 Локальная разработка

```bash
# Установка в режиме разработки
pip install -e .

# Установка инструментов разработки  
pip install black pytest flake8

# Проверка стиля кода
black src/ tests/
flake8 src/ tests/

# Запуск тестов
pytest tests/ --cov=src/
```