# Dockerfile
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y \
    build-essential \
    openmpi-bin libopenmpi-dev \
    openssh-server openssh-client \
    iputils-ping && \
    rm -rf /var/lib/apt/lists/*

# Non-root user for Open MPI (it dislikes running as root)
RUN useradd -m -s /bin/bash mpiuser && \
    echo "mpiuser ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

# SSH daemon setup
RUN mkdir -p /var/run/sshd && \
    printf "PubkeyAuthentication yes\nPasswordAuthentication no\nPermitRootLogin no\n" \
      > /etc/ssh/sshd_config.d/mpi.conf

# Entry script
COPY entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

WORKDIR /work
CMD ["/usr/local/bin/entrypoint.sh"]
