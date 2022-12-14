FROM python:3.8
WORKDIR /code
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
EXPOSE 8000
COPY . .
ENTRYPOINT [ "python" ]
CMD ["app.py"]