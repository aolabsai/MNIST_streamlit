build:
	export DOCKER_BUILDKIT=1
	docker build --secret id=env,src=.env --platform linux/amd64 -t "ao_mnist_app" .
	docker tag ao_mnist_app aolabs/mnist

run:
	docker run -p 8501:8501 aolabs/mnist:latest

push:
	docker push aolabs/mnist:latest
