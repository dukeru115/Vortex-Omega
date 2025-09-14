# 📋 Инструкция по настройке GitHub Actions

## ⚠️ Важно
GitHub Actions workflows не могут быть добавлены через API без специальных разрешений. Вам нужно добавить их вручную через веб-интерфейс GitHub.

## 🚀 Шаги для настройки

### 1. Перейдите в репозиторий на GitHub
Откройте: https://github.com/dukeru115/Vortex-Omega

### 2. Создайте новый workflow
1. Нажмите на вкладку **Actions**
2. Нажмите **New workflow**
3. Выберите **set up a workflow yourself**

### 3. Создайте основной CI/CD workflow

Создайте файл `.github/workflows/main.yml` со следующим содержимым:

```yaml
name: CI/CD Pipeline

on:
  push:
    branches: [ main, develop, genspark_ai_developer ]
  pull_request:
    branches: [ main ]

env:
  PYTHON_VERSION: '3.11'

jobs:
  lint:
    name: 🔍 Lint & Code Quality
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: |
          pip install flake8 black mypy
          black --check src/ tests/
          flake8 src/ tests/ --max-line-length=100

  test:
    name: 🧪 Test
    runs-on: ubuntu-latest
    needs: lint
    services:
      postgres:
        image: postgres:15
        env:
          POSTGRES_PASSWORD: test
        options: >-
          --health-cmd pg_isready
          --health-interval 10s
      redis:
        image: redis:7
        options: >-
          --health-cmd "redis-cli ping"
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: ${{ env.PYTHON_VERSION }}
      - run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
          pytest tests/ --cov=src --cov-report=xml
      - uses: codecov/codecov-action@v3

  build:
    name: 🐋 Build Docker
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ghcr.io
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/build-push-action@v5
        with:
          push: true
          tags: ghcr.io/${{ github.repository }}:latest
```

### 4. Создайте workflow для Docker

Создайте файл `.github/workflows/docker.yml`:

```yaml
name: Docker Image

on:
  push:
    branches: [ main ]
    tags: [ 'v*' ]

env:
  REGISTRY: ghcr.io
  IMAGE_NAME: ${{ github.repository }}

jobs:
  build:
    runs-on: ubuntu-latest
    permissions:
      contents: read
      packages: write
    steps:
      - uses: actions/checkout@v4
      - uses: docker/setup-buildx-action@v3
      - uses: docker/login-action@v3
        with:
          registry: ${{ env.REGISTRY }}
          username: ${{ github.actor }}
          password: ${{ secrets.GITHUB_TOKEN }}
      - uses: docker/metadata-action@v5
        id: meta
        with:
          images: ${{ env.REGISTRY }}/${{ env.IMAGE_NAME }}
      - uses: docker/build-push-action@v5
        with:
          context: .
          push: true
          tags: ${{ steps.meta.outputs.tags }}
          labels: ${{ steps.meta.outputs.labels }}
```

### 5. Создайте workflow для публикации PyPI

Создайте файл `.github/workflows/pypi.yml`:

```yaml
name: Publish to PyPI

on:
  release:
    types: [published]

jobs:
  publish:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.11'
      - run: |
          pip install build twine
          python -m build
      - uses: pypa/gh-action-pypi-publish@release/v1
        with:
          password: ${{ secrets.PYPI_API_TOKEN }}
```

## 🔑 Настройка секретов

В настройках репозитория (Settings → Secrets and variables → Actions) добавьте:

1. `PYPI_API_TOKEN` - токен для публикации в PyPI
2. `CODECOV_TOKEN` - токен для Codecov (опционально)

## ✅ Готово!

После добавления этих файлов через веб-интерфейс GitHub, ваш CI/CD пайплайн будет полностью настроен и автоматически запускаться при push и pull requests.

## 📊 Рекомендуемые сервисы

1. **Codecov** - для отслеживания покрытия кода
   - Зарегистрируйтесь на https://codecov.io
   - Добавьте репозиторий
   - Скопируйте токен в GitHub Secrets

2. **Dependabot** - для автоматического обновления зависимостей
   - Включите в Settings → Security → Dependabot

3. **GitHub Pages** - для публикации документации
   - Включите в Settings → Pages
   - Выберите source: GitHub Actions

## 🎯 Преимущества GitHub Actions

- ✅ Бесплатно для публичных репозиториев
- ✅ Интеграция с GitHub Container Registry
- ✅ Автоматические security updates
- ✅ Матричные сборки для разных версий Python
- ✅ Параллельное выполнение jobs
- ✅ Кеширование зависимостей
- ✅ Встроенная поддержка Docker