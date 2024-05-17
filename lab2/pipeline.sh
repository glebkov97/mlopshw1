pipeline {
    agent any
    stages {
        stage('Setup Python') {
            steps {
                sh '''
                    venv_dir="venv"
                    create_venv() {
                        if [! -d "$venv_dir" ]; then
                            python3 -m venv "$venv_dir"
                            echo "venv created"
                        fi
                    }
                    deactivate_venv() {
                        if [! -z "$VIRTUAL_ENV" ]; then
                            deactivate
                            echo "venv deactivated"
                        fi
                    }
                    activate_venv() {
                        deactivate_venv
                        if [ -d "$venv_dir" ]; then
                            source "$venv_dir/bin/activate"
                            echo "venv activated"
                        else
                            echo "venv not found"
                            return 1
                        fi
                    }
                    install_dependencies() {
                        if [ -f "requirements.txt" ]; then
                            pip install -r requirements.txt -qq
                        else
                            echo "requirements not found"
                            return 1
                        fi
                    }
                    create_venv && activate_venv && install_dependencies
                '''
            }
        }
        stage('Create dataset') {
            steps {
                echo 'Start create dataset'
                sh 'python date_creation.py'
                echo 'Finish create dataset'
            }
        }
        stage('Preprocess data') {
            steps {
                echo 'Start preprocess data'
                sh 'python model_preprocessing.py'
                echo 'Finish preprocess data'
            }
        }
        stage('Train model') {
            steps {
                echo 'Start train model'
                sh 'python model_preparation.py'
                echo 'Finish train model'
            }
        }
        stage('Test model') {
            steps {
                echo 'Start test model'
                sh 'python model_testing.py'
                echo 'Finish test model'
            }
        }
    }
}
