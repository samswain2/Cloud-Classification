# Set python version of docker image
FROM python:3.10.7

# Set working directory of image
WORKDIR /hw02

# Copy necessary files and folders to the image
COPY src/ /hw02/src/
COPY config /hw02/config/
COPY pipeline.py /hw02/
COPY dockerfiles/docker_pipeline/requirements.txt /hw02/requirements.txt
COPY tests/ /hw02/tests/

# Install dependencies 
RUN pip install --trusted-host pypi.python.org -r /hw02/requirements.txt

# Run pipeline.py when container launches
CMD [ "python", "/hw02/tests/unit_tests.py"]