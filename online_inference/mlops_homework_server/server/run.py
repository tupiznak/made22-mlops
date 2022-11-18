import uvicorn


def main():
    uvicorn.run('mlops_homework_server.server.main:app', port=34222)


if __name__ == '__main__':
    main()
