if [[ -z "${PROJECT_PATH}" ]]; then
  echo "Need env PROJECT_PATH. exit"
  exit 1
fi

mkdir -p ./dags ./logs ./plugins ./data
echo -e "AIRFLOW_UID=$(id -u)" >> .env
sudo chmod 666 /var/run/docker.sock
docker-compose -f docker-compose.yaml up -d