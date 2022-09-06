FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7

WORKDIR /

RUN pip install -U pip

COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py /main.py

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]

EXPOSE 8080
