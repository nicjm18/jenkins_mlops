pipeline {
  agent any
  options { timestamps(); ansiColor('xterm') }

  stages {
    stage('Checkout') {
      steps { checkout scm }
    }

    stage('Python venv') {
      steps {
        sh '''
          python3 -V
          python3 -m venv .venv
          . .venv/bin/activate
          pip install --upgrade pip
          pip install -r requirements.txt || true  # si está vacío, no romper
        '''
      }
    }

    stage('PyOps Checks') {
      steps {
        sh '''
          . .venv/bin/activate
          python -m pyops.validate reports
        '''
      }
    }
  }

  post {
    always {
      junit testResults: 'reports/junit.xml', allowEmptyResults: true
      archiveArtifacts artifacts: 'reports/**', fingerprint: true
    }
    success { echo '✅ Validaciones OK' }
    failure { echo '❌ Falló alguna validación. Revisa Test Results y reports/junit.xml' }
  }
}
