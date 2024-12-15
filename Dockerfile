FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Configurar variables de entorno para evitar preguntas interactivas
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=America/Mexico_City

WORKDIR /app

# Instalar Python y otras dependencias del sistema
RUN apt-get update && apt-get install -y \
    software-properties-common \
    tzdata \
    && ln -fs /usr/share/zoneinfo/$TZ /etc/localtime \
    && dpkg-reconfigure -f noninteractive tzdata \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-distutils \
    python3.9-dev \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear enlace simbólico para python3
RUN ln -s /usr/bin/python3.9 /usr/local/bin/python3 \
    && ln -s /usr/bin/python3.9 /usr/local/bin/python

# Actualizar pip
RUN python3 -m pip install --upgrade pip

# Instalar PyTorch primero y por separado
RUN pip3 install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu118

# Instalar poetry
RUN pip3 install poetry

# Copiar solo pyproject.toml primero
COPY pyproject.toml ./

# Modificar pyproject.toml para excluir torch (ya que lo instalamos manualmente)
RUN sed -i '/torch/d' pyproject.toml

# Instalar dependencias con poetry
RUN poetry config virtualenvs.create false \
    && poetry install --no-interaction --no-ansi

# Instalar el resto de paquetes críticos
RUN pip3 install --no-cache-dir \
    transformers \
    accelerate>=0.26.0

# Copiar el código
COPY ./app ./app

# Comando para ejecutar
CMD ["poetry", "run", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]