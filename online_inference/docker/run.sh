cd ../..
docker run --name made22-mlops -p34222:34222 -eKAGGLE_USERNAME="<name>" -eKAGGLE_KEY="<key>" -v"<storage>":/s3_storage --rm ajdioawd21e/mlops:1.0