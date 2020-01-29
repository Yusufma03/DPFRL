uid=`id -u`
( echo '### DO NOT EDIT DIRECTLY, SEE Dockerfile.template ###'; sed -e "s/<<UID>>/${uid}/" < Dockerfile.cuda.template ) > Dockerfile
if hash nvidia-docker 2>/dev/null; then
    nvidia-docker build -t yusufma/pfrl .
else
    docker build -t  yusufma/pfrl .
fi
