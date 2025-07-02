FROM carlasim/carla:0.9.15

ENV DEBIAN_FRONTEND=noninteractive

USER root

RUN apt-get update && apt-get install -y \
    mesa-utils \
    libgl1-mesa-glx \
    libgl1-mesa-dri \
    xdg-user-dirs \
    xdg-utils \
    strace \
    xvfb \
    x11-utils \
    x11-xserver-utils \
    libgl1-mesa-glx \
    libglu1-mesa \
    libsdl2-2.0-0 \
    libxrandr2 \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libx11-6 \
    libxxf86vm1 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

USER carla

EXPOSE 8501-8502

# SDL_VIDEODRIVER=offscreen SDL_HINT_CUDA_DEVICE=0 ./CarlaUE4.sh -RenderOffScreen -opengl -ResX=800 -ResY=600 -nosound