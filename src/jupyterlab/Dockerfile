FROM jupyter/minimal-notebook

ENV LANG=C.UTF-8

# Install Python Packages & Requirements (Done near end to avoid invalidating cache)
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Expose Jupyter port & cmd
EXPOSE 8888
USER root
RUN apt-get update -yy && apt-get install ffmpeg -y
RUN mkdir -p /opt/app/data
ENTRYPOINT jupyter lab --ip=* --port=8888 --no-browser --notebook-dir=/opt/app/data --allow-root
