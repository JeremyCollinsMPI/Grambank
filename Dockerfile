FROM tensorflow/tensorflow:1.12.0-py3
RUN pip install python-nexus geocoder 
RUN pip install --upgrade pandas
