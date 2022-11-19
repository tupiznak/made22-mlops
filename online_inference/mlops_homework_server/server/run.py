import uvicorn


def main():
    uvicorn.run('mlops_homework_server.server.main:app', port=34222, host='0.0.0.0')


if __name__ == '__main__':
    main()
