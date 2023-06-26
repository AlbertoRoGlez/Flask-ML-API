FROM python:3.8.13 as base

ARG UID=1000
ARG GID=1000

# Add a non-root user
RUN addgroup --gid $GID model
RUN adduser --disabled-login --geco '' --uid $UID --gid $GID model
USER model

# Append /home/app/.local/bin/ to PATH variable because
# gunicorn is installed there.
ENV PYTHONPATH=$PYTHONPATH:/home/
ENV PATH=$PATH:/home/model/.local/bin/

ADD requirements.txt .
RUN pip3 install -r requirements.txt

ENV PYTHONPATH=$PYTHONPATH:/home/model/

COPY . /home/model/

WORKDIR /home/model/

FROM base as test
RUN ["pytest", "-v", "tests/"]

FROM base as build
ENTRYPOINT ["python3", "ml_service.py"]

#----------------------

# FROM python:3.8.13 as base

# ARG UID=1000
# ARG GID=1000

# # Add a non-root user
# RUN addgroup --gid $GID app
# RUN adduser --disabled-login --geco '' --uid $UID --gid $GID app
# USER app

# # Append /home/app/.local/bin/ to PATH variable because
# # gunicorn is installed there.
# ENV PYTHONPATH=$PYTHONPATH:/src/
# ENV PATH=$PATH:/home/app/.local/bin/

# ADD requirements.txt .
# RUN pip3 install -r requirements.txt

# ENV PYTHONPATH=$PYTHONPATH:/src/

# COPY ./ /src/

# WORKDIR /src

# FROM base as test
# RUN ["pytest", "-v", "/src/tests"]

# FROM base as build
# ENTRYPOINT ["python3", "/src/ml_service.py"]