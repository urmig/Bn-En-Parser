FROM python:3.6-slim
COPY ./StackedParserClass.py /deploy/
COPY ./ParserClass.py /deploy/
COPY ./utils.py /deploy/
COPY ./static/css/style.css /deploy/static/css/
COPY ./templates/index.html /deploy/templates/
COPY ./lib/pseudoProjectivity.py /deploy/lib/
COPY ./lib/arc_eager.py /deploy/lib/
COPY ./app.py /deploy/
COPY ./requirements.txt /deploy/
COPY ./bn-en-stacked.model* /deploy/
WORKDIR /deploy/

RUN pip3 install -r requirements.txt && \
    pip3 install flask  
EXPOSE 80
ENTRYPOINT ["python", "app.py"]
