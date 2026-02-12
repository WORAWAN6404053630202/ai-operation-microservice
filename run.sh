#!/bin/bash
export $(grep -v '^#' env.properties | xargs)

if [ "$1" = "prod" ]; then
    # Start the service in production mode
    docker-compose -f ./docker-compose.prod.yaml up --build
elif [ "$1" = "sg" ]; then
    # Start the service in nolive mode
    docker-compose -f ./docker-compose.staging.yaml up --build
else
    # Start the service in development mode
    docker-compose -f ./docker-compose.dev.yaml up $1 $2
fi