FROM python:3.9.13-slim AS python

COPY . ./

RUN pip3 install -r requirements.txt

RUN pip3 install torch --extra-index-url https://download.pytorch.org/whl/cpu

CMD ["python", "app.py"]