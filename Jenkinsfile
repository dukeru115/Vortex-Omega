// Jenkins Pipeline для Vortex-Omega NFCS
// Декларативный пайплайн с параллельным выполнением

@Library('shared-library') _

pipeline {
    agent any
    
    // Параметры сборки
    parameters {
        choice(
            name: 'ENVIRONMENT',
            choices: ['development', 'staging', 'production'],
            description: 'Окружение для развёртывания'
        )
        booleanParam(
            name: 'RUN_PERFORMANCE_TESTS',
            defaultValue: false,
            description: 'Запустить тесты производительности'
        )
        booleanParam(
            name: 'DEPLOY_AFTER_BUILD',
            defaultValue: true,
            description: 'Автоматически развернуть после успешной сборки'
        )
    }
    
    // Переменные окружения
    environment {
        DOCKER_REGISTRY = 'registry.example.com'
        DOCKER_IMAGE = "${DOCKER_REGISTRY}/vortex-omega"
        PYTHON_VERSION = '3.11'
        SLACK_CHANNEL = '#vortex-omega-ci'
        SONARQUBE_SERVER = 'SonarQube'
    }
    
    // Опции пайплайна
    options {
        timestamps()
        timeout(time: 1, unit: 'HOURS')
        buildDiscarder(logRotator(numToKeepStr: '10'))
        disableConcurrentBuilds()
        skipDefaultCheckout()
    }
    
    // Триггеры
    triggers {
        pollSCM('H/5 * * * *')
        githubPush()
    }
    
    stages {
        stage('🎬 Initialization') {
            steps {
                script {
                    echo "🚀 Начало сборки #${BUILD_NUMBER}"
                    echo "📦 Ветка: ${BRANCH_NAME}"
                    echo "🎯 Окружение: ${params.ENVIRONMENT}"
                }
                
                checkout scm
                
                script {
                    // Получаем информацию о коммите
                    env.GIT_COMMIT_MSG = sh(
                        script: 'git log -1 --pretty=%B',
                        returnStdout: true
                    ).trim()
                    env.GIT_AUTHOR = sh(
                        script: 'git log -1 --pretty=%ae',
                        returnStdout: true
                    ).trim()
                }
            }
        }
        
        stage('🔍 Code Quality') {
            parallel {
                stage('Python Linting') {
                    steps {
                        sh '''
                            echo "🐍 Проверка качества кода Python..."
                            python -m venv venv
                            . venv/bin/activate
                            pip install flake8 black pylint mypy
                            
                            black --check src/ tests/
                            flake8 src/ tests/ --max-line-length=100
                            pylint src/ --fail-under=8.0
                            mypy src/ --ignore-missing-imports
                        '''
                    }
                }
                
                stage('Security Scan') {
                    steps {
                        sh '''
                            echo "🔐 Сканирование безопасности..."
                            . venv/bin/activate
                            pip install bandit safety
                            
                            bandit -r src/ -f json -o bandit-report.json
                            safety check -r requirements.txt --json
                        '''
                    }
                }
                
                stage('SonarQube Analysis') {
                    when {
                        branch 'main'
                    }
                    steps {
                        script {
                            withSonarQubeEnv('SonarQube') {
                                sh '''
                                    sonar-scanner \
                                        -Dsonar.projectKey=vortex-omega \
                                        -Dsonar.sources=src \
                                        -Dsonar.tests=tests \
                                        -Dsonar.python.coverage.reportPaths=coverage.xml
                                '''
                            }
                        }
                    }
                }
            }
        }
        
        stage('🧪 Testing') {
            parallel {
                stage('Unit Tests') {
                    steps {
                        sh '''
                            echo "🧪 Запуск юнит-тестов..."
                            . venv/bin/activate
                            pip install -r requirements.txt
                            pip install -r requirements-dev.txt
                            
                            pytest tests/unit/ \
                                -v \
                                --cov=src \
                                --cov-report=xml \
                                --cov-report=html \
                                --junitxml=test-results/junit.xml
                        '''
                    }
                    post {
                        always {
                            junit 'test-results/junit.xml'
                            publishHTML([
                                reportDir: 'htmlcov',
                                reportFiles: 'index.html',
                                reportName: 'Coverage Report'
                            ])
                        }
                    }
                }
                
                stage('Integration Tests') {
                    steps {
                        sh '''
                            echo "🔗 Запуск интеграционных тестов..."
                            docker-compose -f docker-compose.test.yml up -d
                            sleep 30
                            
                            . venv/bin/activate
                            pytest tests/integration/ -v --junitxml=test-results/integration.xml
                            
                            docker-compose -f docker-compose.test.yml down -v
                        '''
                    }
                }
                
                stage('Performance Tests') {
                    when {
                        expression { params.RUN_PERFORMANCE_TESTS == true }
                    }
                    steps {
                        sh '''
                            echo "⚡ Запуск тестов производительности..."
                            . venv/bin/activate
                            pip install locust pytest-benchmark
                            
                            pytest tests/performance/ \
                                --benchmark-only \
                                --benchmark-json=benchmark.json
                            
                            locust -f tests/locustfile.py \
                                --headless \
                                --users 100 \
                                --spawn-rate 10 \
                                --run-time 60s \
                                --host http://localhost:8080
                        '''
                    }
                }
            }
        }
        
        stage('🏗️ Build') {
            steps {
                script {
                    echo "🐋 Сборка Docker образа..."
                    
                    def imageTag = "${env.BUILD_NUMBER}-${env.GIT_COMMIT[0..7]}"
                    
                    docker.build("${DOCKER_IMAGE}:${imageTag}")
                    docker.build("${DOCKER_IMAGE}:latest")
                    
                    if (env.BRANCH_NAME == 'main') {
                        docker.withRegistry("https://${DOCKER_REGISTRY}", 'docker-credentials') {
                            docker.image("${DOCKER_IMAGE}:${imageTag}").push()
                            docker.image("${DOCKER_IMAGE}:latest").push()
                        }
                    }
                }
            }
        }
        
        stage('🔐 Security Validation') {
            parallel {
                stage('Container Scan') {
                    steps {
                        sh """
                            echo "🔍 Сканирование Docker образа..."
                            trivy image \
                                --severity HIGH,CRITICAL \
                                --format json \
                                --output trivy-report.json \
                                ${DOCKER_IMAGE}:latest
                        """
                    }
                }
                
                stage('OWASP Dependency Check') {
                    steps {
                        dependencyCheck additionalArguments: '''
                            --scan src/
                            --format JSON
                            --out dependency-check-report.json
                        ''', odcInstallation: 'OWASP-DC'
                        
                        dependencyCheckPublisher pattern: 'dependency-check-report.json'
                    }
                }
            }
        }
        
        stage('🚀 Deploy') {
            when {
                expression { params.DEPLOY_AFTER_BUILD == true }
            }
            stages {
                stage('Deploy to Staging') {
                    when {
                        expression { params.ENVIRONMENT == 'staging' }
                    }
                    steps {
                        script {
                            echo "🌐 Развёртывание на staging..."
                            
                            sshagent(['staging-ssh-key']) {
                                sh '''
                                    ssh user@staging.example.com << EOF
                                        cd /opt/vortex-omega
                                        docker-compose pull
                                        docker-compose up -d --force-recreate
                                        docker-compose ps
                                    EOF
                                '''
                            }
                        }
                    }
                }
                
                stage('Deploy to Production') {
                    when {
                        allOf {
                            expression { params.ENVIRONMENT == 'production' }
                            branch 'main'
                        }
                    }
                    input {
                        message "Развернуть в production?"
                        ok "Развернуть"
                        submitter "DevOps,Admin"
                    }
                    steps {
                        script {
                            echo "🚀 Развёртывание в production..."
                            
                            // Blue-Green deployment
                            sh '''
                                kubectl set image deployment/vortex-omega-blue \
                                    vortex-omega=${DOCKER_IMAGE}:${BUILD_NUMBER} \
                                    --namespace=production
                                
                                kubectl rollout status deployment/vortex-omega-blue \
                                    --namespace=production
                                
                                # Переключение трафика
                                kubectl patch service vortex-omega \
                                    -p '{"spec":{"selector":{"version":"blue"}}}' \
                                    --namespace=production
                            '''
                        }
                    }
                }
            }
        }
        
        stage('🧪 Smoke Tests') {
            when {
                expression { params.DEPLOY_AFTER_BUILD == true }
            }
            steps {
                script {
                    def targetUrl = params.ENVIRONMENT == 'production' ? 
                        'https://vortex-omega.example.com' : 
                        'https://staging.vortex-omega.example.com'
                    
                    sh """
                        echo "🔥 Запуск smoke тестов..."
                        curl -f ${targetUrl}/health || exit 1
                        curl -f ${targetUrl}/metrics || exit 1
                        
                        . venv/bin/activate
                        pytest tests/smoke/ -v --base-url=${targetUrl}
                    """
                }
            }
        }
    }
    
    post {
        always {
            echo "🧹 Очистка ресурсов..."
            sh 'docker-compose -f docker-compose.test.yml down -v || true'
            
            archiveArtifacts artifacts: '''
                **/test-results/*.xml,
                **/coverage.xml,
                **/*-report.json,
                **/benchmark.json
            ''', allowEmptyArchive: true
            
            cleanWs()
        }
        
        success {
            script {
                slackSend(
                    channel: env.SLACK_CHANNEL,
                    color: 'good',
                    message: """
                        ✅ Сборка #${BUILD_NUMBER} успешна!
                        Проект: ${JOB_NAME}
                        Ветка: ${BRANCH_NAME}
                        Автор: ${GIT_AUTHOR}
                        Сообщение: ${GIT_COMMIT_MSG}
                    """
                )
            }
        }
        
        failure {
            script {
                slackSend(
                    channel: env.SLACK_CHANNEL,
                    color: 'danger',
                    message: """
                        ❌ Сборка #${BUILD_NUMBER} провалилась!
                        Проект: ${JOB_NAME}
                        Ветка: ${BRANCH_NAME}
                        Автор: ${GIT_AUTHOR}
                        Проверьте логи: ${BUILD_URL}
                    """
                )
            }
        }
        
        unstable {
            script {
                slackSend(
                    channel: env.SLACK_CHANNEL,
                    color: 'warning',
                    message: """
                        ⚠️ Сборка #${BUILD_NUMBER} нестабильна
                        Проект: ${JOB_NAME}
                        Проверьте результаты тестов
                    """
                )
            }
        }
    }
}