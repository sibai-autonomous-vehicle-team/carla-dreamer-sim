FROM carlasim/carla:0.9.15
USER root

RUN apt-get update && \
    apt-get install -y \
        xdg-user-dirs \
        xdg-utils \
        strace \
        xvfb \
        x11-utils \
        x11-xserver-utils \
        mesa-utils && \
    apt-get clean


USER carla