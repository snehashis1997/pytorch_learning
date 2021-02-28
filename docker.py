# day 1

docker run <container name>
docker ps
docker ps -a
docker stop <container name>
docker rm <container name>
docker images 
docker rmi <container name>
docker pull <container name>
docker pull ubuntu
docker run <container name> sleep 5
docker run <web app container name>
docker run -d <web app container name>
docker attach <web app container name>
docker run redis:4.0
docker run -i redis:4.0
docker run -it redis:4.0
docker run -p 80:5000 <web app container name>
docker run -v /opt/datadir:/var/lib/mysql mysql
docker inspect <container>
docker logs <container>

export APP_COLOR=blue; python app.py
docker run -e APP_COLOR=blue redis:4.0
docker inspect <container>


# day 2







